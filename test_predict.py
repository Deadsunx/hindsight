#!/usr/bin/env python3
"""Offline tests for the grading half of Hindsight.

The fetching half is allowed to fail — a dead API just means no call that day.
The *grading* half is not: if it is wrong, the public track record is a lie.
So everything below runs on synthetic archives with no network access.
"""

import math
import os
import unittest
from unittest import mock

import predict


def day(date, observations=None, predictions=None):
    return {
        "date": date,
        "fetched_at": f"{date}T04:00:00+00:00",
        "observations": observations or {},
        "predictions": predictions or [],
    }


def bet(pid, date, call, confidence=0.7, basis=None, horizon=1):
    return {
        "id": pid,
        "question": predict.PREDICTORS[pid]["question"],
        "rule": predict.PREDICTORS[pid]["rule"],
        "call": call,
        "confidence": confidence,
        "basis": basis or {},
        "made_on": date,
        "resolve_on": predict.shift(date, horizon),
    }


class TestEnv(unittest.TestCase):
    """GitHub Actions passes an unset repo variable as "", not as absent.

    A plain os.environ.get() would then hand back a blank latitude and build a
    URL that 400s — which is exactly how the first CI run lost the weather bet.
    """

    def test_missing_falls_back(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(predict.env("CITY_TZ", "Asia/Kolkata"), "Asia/Kolkata")

    def test_empty_falls_back(self):
        with mock.patch.dict(os.environ, {"CITY_TZ": ""}):
            self.assertEqual(predict.env("CITY_TZ", "Asia/Kolkata"), "Asia/Kolkata")

    def test_whitespace_falls_back(self):
        with mock.patch.dict(os.environ, {"CITY_LAT": "   "}):
            self.assertEqual(predict.env("CITY_LAT", "28.6139"), "28.6139")

    def test_real_value_wins(self):
        with mock.patch.dict(os.environ, {"CITY_NAME": "Greater Noida"}):
            self.assertEqual(predict.env("CITY_NAME", "New Delhi"), "Greater Noida")


class TestHelpers(unittest.TestCase):
    def test_shift(self):
        self.assertEqual(predict.shift("2026-01-31", 1), "2026-02-01")
        self.assertEqual(predict.shift("2026-03-01", -1), "2026-02-28")

    def test_obs_of_ignores_failed_sources(self):
        days = {"2026-01-01": day("2026-01-01", {"btc": {"error": "timeout"}})}
        self.assertIsNone(predict.obs_of(days, "2026-01-01", "btc"))

    def test_obs_of_missing_day(self):
        self.assertIsNone(predict.obs_of({}, "2026-01-01", "btc"))


class TestResolvers(unittest.TestCase):
    def test_btc_up_and_down(self):
        days = {
            "2026-01-01": day("2026-01-01", {"btc": {"usd": 60000}}),
            "2026-01-02": day("2026-01-02", {"btc": {"usd": 61000}}),
        }
        pred = bet("btc_up", "2026-01-01", True)
        self.assertTrue(predict.resolve_btc(pred, days))

        days["2026-01-02"]["observations"]["btc"]["usd"] = 59000
        self.assertFalse(predict.resolve_btc(pred, days))

    def test_btc_void_when_next_day_missing(self):
        days = {"2026-01-01": day("2026-01-01", {"btc": {"usd": 60000}})}
        self.assertIsNone(predict.resolve_btc(bet("btc_up", "2026-01-01", True), days))

    def test_kp_threshold_is_strict(self):
        days = {"2026-01-02": day("2026-01-02", {"kp": {"max_kp": 3.0}})}
        self.assertFalse(predict.resolve_kp(bet("kp_storm", "2026-01-01", True), days))
        days["2026-01-02"]["observations"]["kp"]["max_kp"] = 3.33
        self.assertTrue(predict.resolve_kp(bet("kp_storm", "2026-01-01", True), days))

    def test_quake_threshold_is_inclusive(self):
        days = {"2026-01-02": day("2026-01-02", {"quakes": {"max_mag": 5.0}})}
        self.assertTrue(predict.resolve_quake(bet("quake_m5", "2026-01-01", True), days))
        days["2026-01-02"]["observations"]["quakes"]["max_mag"] = 4.9
        self.assertFalse(predict.resolve_quake(bet("quake_m5", "2026-01-01", True), days))

    def test_wiki_hold(self):
        days = {
            "2026-01-01": day("2026-01-01", {"wiki": {"top": "Dune"}}),
            "2026-01-02": day("2026-01-02", {"wiki": {"top": "Dune"}}),
        }
        pred = bet("wiki_hold", "2026-01-01", True)
        self.assertTrue(predict.resolve_wiki(pred, days))
        days["2026-01-02"]["observations"]["wiki"]["top"] = "Cleopatra"
        self.assertFalse(predict.resolve_wiki(pred, days))

    def test_hn_fade(self):
        pred = bet("hn_fade", "2026-01-01", True, basis={"top_id": 42})
        days = {"2026-01-02": day("2026-01-02", {"hn": {"top10": [1, 2, 3]}})}
        self.assertTrue(predict.resolve_hn(pred, days))  # 42 is gone -> faded
        days["2026-01-02"]["observations"]["hn"]["top10"] = [1, 42, 3]
        self.assertFalse(predict.resolve_hn(pred, days))

    def test_forecast_within_tolerance(self):
        pred = bet(
            "forecast_hit", "2026-01-01", True,
            basis={"forecast_c": 30.0, "for_date": "2026-01-02"}, horizon=2,
        )
        days = {
            "2026-01-01": day("2026-01-01"),
            "2026-01-03": day(
                "2026-01-03",
                {"weather": {"actual_prev": 31.5, "actual_prev_date": "2026-01-02"}},
            ),
        }
        self.assertTrue(predict.resolve_forecast(pred, days))
        days["2026-01-03"]["observations"]["weather"]["actual_prev"] = 35.0
        self.assertFalse(predict.resolve_forecast(pred, days))

    def test_forecast_settles_from_a_late_run(self):
        """A missed cron day must not void the bet — any later run carrying the
        completed high for that date settles it."""
        pred = bet(
            "forecast_hit", "2026-01-01", True,
            basis={"forecast_c": 30.0, "for_date": "2026-01-02"}, horizon=2,
        )
        days = {
            "2026-01-01": day("2026-01-01"),
            "2026-01-05": day(
                "2026-01-05",
                {"weather": {"actual_prev": 30.5, "actual_prev_date": "2026-01-02"}},
            ),
        }
        self.assertTrue(predict.resolve_forecast(pred, days))

    def test_forecast_ignores_the_wrong_date(self):
        pred = bet(
            "forecast_hit", "2026-01-01", True,
            basis={"forecast_c": 30.0, "for_date": "2026-01-02"}, horizon=2,
        )
        days = {
            "2026-01-01": day("2026-01-01"),
            "2026-01-04": day(
                "2026-01-04",
                {"weather": {"actual_prev": 30.1, "actual_prev_date": "2026-01-03"}},
            ),
        }
        self.assertIsNone(predict.resolve_forecast(pred, days))


class TestGrading(unittest.TestCase):
    def _archive(self):
        return {
            "2026-01-01": day(
                "2026-01-01",
                {"btc": {"usd": 60000}, "quakes": {"max_mag": 6.1}},
                [
                    bet("btc_up", "2026-01-01", True, 0.55),
                    bet("quake_m5", "2026-01-01", False, 0.70),
                ],
            ),
            "2026-01-02": day(
                "2026-01-02", {"btc": {"usd": 61000}, "quakes": {"max_mag": 4.2}}
            ),
        }

    def test_grades_both_directions(self):
        grades = predict.grade_all(self._archive())
        by_id = {g["id"]: g for g in grades}
        self.assertTrue(by_id["btc_up"]["correct"])  # called up, went up
        self.assertTrue(by_id["quake_m5"]["correct"])  # called no, none happened
        self.assertFalse(by_id["quake_m5"]["actual"])

    def test_a_wrong_call_is_recorded_as_wrong(self):
        days = self._archive()
        days["2026-01-02"]["observations"]["btc"]["usd"] = 50000
        grades = {g["id"]: g for g in predict.grade_all(days)}
        self.assertFalse(grades["btc_up"]["correct"])

    def test_is_idempotent(self):
        days = self._archive()
        self.assertEqual(predict.grade_all(days), predict.grade_all(days))

    def test_unknown_predictor_ids_are_skipped(self):
        days = self._archive()
        days["2026-01-01"]["predictions"].append(
            {
                "id": "retired_rule", "call": True, "confidence": 0.9,
                "made_on": "2026-01-01", "resolve_on": "2026-01-02", "basis": {},
            }
        )
        self.assertEqual(len(predict.grade_all(days)), 2)

    def test_unsettled_predictions_are_absent(self):
        days = {
            "2026-01-01": day(
                "2026-01-01", {"btc": {"usd": 60000}},
                [bet("btc_up", "2026-01-01", True)],
            )
        }
        self.assertEqual(predict.grade_all(days), [])


class TestLedger(unittest.TestCase):
    def _graded(self):
        days = {
            "2026-01-01": day(
                "2026-01-01",
                {"btc": {"usd": 60000}, "quakes": {"max_mag": 6.1}},
                [
                    bet("btc_up", "2026-01-01", True, 0.80),
                    bet("quake_m5", "2026-01-01", False, 0.70),
                ],
            ),
            "2026-01-02": day(
                "2026-01-02", {"btc": {"usd": 61000}, "quakes": {"max_mag": 4.2}}
            ),
        }
        return days, predict.grade_all(days)

    def test_accuracy(self):
        days, grades = self._graded()
        ledger = predict.build_ledger(days, grades, "2026-01-02")
        self.assertEqual(ledger["totals"]["graded"], 2)
        self.assertEqual(ledger["totals"]["correct"], 2)
        self.assertEqual(ledger["totals"]["accuracy"], 1.0)

    def test_brier_uses_probability_of_the_event(self):
        # yes @ 0.80, happened  -> (0.80 - 1)^2 = 0.04
        # no  @ 0.70, did not   -> (0.30 - 0)^2 = 0.09   (p_event = 1 - 0.70)
        days, grades = self._graded()
        ledger = predict.build_ledger(days, grades, "2026-01-02")
        self.assertAlmostEqual(ledger["totals"]["brier"], 0.065, places=4)

    def test_calibration_buckets_by_confidence(self):
        days, grades = self._graded()
        ledger = predict.build_ledger(days, grades, "2026-01-02")
        buckets = {b["label"]: b for b in ledger["calibration"]}
        self.assertEqual(buckets["70–80%"]["n"], 1)
        self.assertEqual(buckets["80–90%"]["n"], 1)
        self.assertEqual(buckets["50–60%"]["n"], 0)
        self.assertIsNone(buckets["50–60%"]["actual"])

    def test_open_versus_void(self):
        days = {
            "2026-01-01": day(  # never settled, and the day has passed -> void
                "2026-01-01", {"btc": {"usd": 60000}},
                [bet("btc_up", "2026-01-01", True)],
            ),
            "2026-01-05": day(  # placed today, resolves tomorrow -> open
                "2026-01-05", {"btc": {"usd": 60000}},
                [bet("btc_up", "2026-01-05", True)],
            ),
        }
        ledger = predict.build_ledger(days, predict.grade_all(days), "2026-01-05")
        self.assertEqual(ledger["totals"]["open"], 1)
        self.assertEqual(ledger["totals"]["void"], 1)

    def test_empty_archive_does_not_divide_by_zero(self):
        ledger = predict.build_ledger({}, [], "2026-01-01")
        self.assertIsNone(ledger["totals"]["accuracy"])
        self.assertIsNone(ledger["totals"]["brier"])
        self.assertIsNone(ledger["totals"]["log_loss"])
        self.assertEqual(ledger["totals"]["graded"], 0)

    def test_baselines_travel_with_the_ledger(self):
        days, grades = self._graded()
        ledger = predict.build_ledger(days, grades, "2026-01-02")
        coin = ledger["baselines"]["coin_flip"]
        self.assertEqual(coin["n"], 2)
        self.assertEqual(coin["brier"], 0.25)
        self.assertEqual(ledger["baselines"]["base_rate"]["n"], 2)

    def test_every_predictor_row_carries_its_sample_size(self):
        days, grades = self._graded()
        ledger = predict.build_ledger(days, grades, "2026-01-02")
        for pid, row in ledger["by_predictor"].items():
            self.assertIn("graded", row, pid)
            self.assertIn("brier", row, pid)
            self.assertIn("log_loss", row, pid)
            if not row["graded"]:
                self.assertIsNone(row["brier"], pid)
                self.assertIsNone(row["log_loss"], pid)


class TestScoring(unittest.TestCase):
    """The metrics themselves. If these are wrong the whole record is wrong."""

    def _g(self, call, confidence, actual):
        return {
            "call": call, "confidence": confidence,
            "actual": actual, "correct": call == actual,
        }

    def test_p_event_inverts_a_no_call(self):
        # Confidence is stated in the call, not the event: a 70% "no" is a 30%
        # chance of the event happening.
        self.assertAlmostEqual(predict.p_event(self._g(False, 0.7, False)), 0.3)
        self.assertAlmostEqual(predict.p_event(self._g(True, 0.7, False)), 0.7)

    def test_brier_matches_hand_arithmetic(self):
        # yes @ 0.80, happened -> (0.8 - 1)^2 = 0.04
        # no  @ 0.70, did not  -> (0.3 - 0)^2 = 0.09
        marks = predict.score([self._g(True, 0.80, True), self._g(False, 0.70, False)])
        self.assertAlmostEqual(marks["brier"], 0.065, places=4)

    def test_log_loss_punishes_the_confident_miss_harder_than_brier(self):
        """The whole reason log loss is reported alongside accuracy."""
        timid = predict.score([self._g(True, 0.55, False)])
        brash = predict.score([self._g(True, 0.95, False)])
        # Both are simply "wrong" to accuracy — it cannot tell them apart.
        self.assertEqual(timid["accuracy"], brash["accuracy"])
        self.assertGreater(brash["brier"], timid["brier"])
        # Log loss opens a far wider gap on the same two bets.
        self.assertGreater(
            brash["log_loss"] - timid["log_loss"],
            brash["brier"] - timid["brier"],
        )

    def test_a_certain_wrong_call_does_not_produce_infinity(self):
        marks = predict.score([self._g(True, 1.0, False)])
        self.assertTrue(math.isfinite(marks["log_loss"]))
        self.assertGreater(marks["log_loss"], 30)

    def test_coin_flip_is_the_reference_every_score_must_beat(self):
        self.assertEqual(predict.coin_flip(10)["brier"], 0.25)
        self.assertAlmostEqual(predict.coin_flip(10)["log_loss"], math.log(2), places=4)
        self.assertIsNone(predict.coin_flip(0)["brier"])

    def test_base_rate_exposes_an_unpredictable_question(self):
        """A question whose event fires 90% of the time makes 90% accuracy
        worthless — always calling yes would have matched it."""
        grades = [self._g(True, 0.9, True)] * 9 + [self._g(True, 0.9, False)]
        rate = predict.base_rate(grades)
        self.assertAlmostEqual(rate["rate"], 0.9)
        self.assertAlmostEqual(rate["majority_accuracy"], 0.9)

    def test_empty_scores_are_none_not_zero(self):
        """Zero would read as a perfect Brier. None reads as no evidence."""
        marks = predict.score([])
        self.assertEqual(marks["n"], 0)
        self.assertIsNone(marks["brier"])
        self.assertIsNone(marks["log_loss"])
        self.assertIsNone(marks["accuracy"])


class TestPredictors(unittest.TestCase):
    def test_every_predictor_declares_a_question_and_horizon(self):
        for pid, spec in predict.PREDICTORS.items():
            self.assertTrue(spec["question"], pid)
            self.assertIn(spec["horizon"], (1, 2), pid)
            self.assertTrue(callable(spec["make"]), pid)
            self.assertTrue(callable(spec["resolve"]), pid)

    def test_confidence_is_never_below_a_coin_flip(self):
        obs = {
            "btc": {"usd": 60000, "usd_24h_change": -3.0},
            "kp": {"max_kp": 1.0},
            "quakes": {"max_mag": 6.0},
            "wiki": {"top": "Dune"},
            "hn": {"top_id": 7, "top10": [7]},
            "weather": {"forecast_next": 30.0, "forecast_next_date": "2026-01-02"},
        }
        for pid, spec in predict.PREDICTORS.items():
            if spec["variant"] != "rule":
                continue  # models need settled history; covered in TestLearning
            result = spec["make"](obs, {}, "2026-01-01")
            self.assertIsNotNone(result, pid)
            _, confidence, _ = result
            self.assertGreaterEqual(confidence, 0.5, pid)
            self.assertLessEqual(confidence, 0.95, pid)

    def test_predictors_return_none_without_data(self):
        for pid, spec in predict.PREDICTORS.items():
            self.assertIsNone(spec["make"]({}, {}, "2026-01-01"), pid)

    def test_every_family_has_both_a_rule_and_a_model(self):
        families = {}
        for spec in predict.PREDICTORS.values():
            families.setdefault(spec["family"], set()).add(spec["variant"])
        self.assertEqual(len(families), 6)
        for family, variants in families.items():
            self.assertEqual(variants, {"rule", "model"}, family)

    def test_variants_share_a_resolver(self):
        """Both variants answer the same question, so one settles both."""
        for family in [s["family"] for s in predict.PREDICTORS.values()
                       if s["variant"] == "rule"]:
            self.assertIs(
                predict.PREDICTORS[family]["resolve"],
                predict.PREDICTORS[f"{family}_ml"]["resolve"],
                family,
            )
            self.assertEqual(
                predict.PREDICTORS[family]["horizon"],
                predict.PREDICTORS[f"{family}_ml"]["horizon"],
                family,
            )

    def test_momentum_follows_the_move(self):
        up = predict.make_btc({"btc": {"usd": 1, "usd_24h_change": 2.0}}, {}, "2026-01-01")
        down = predict.make_btc({"btc": {"usd": 1, "usd_24h_change": -2.0}}, {}, "2026-01-01")
        self.assertTrue(up[0])
        self.assertFalse(down[0])

    def test_wiki_is_more_confident_when_already_streaking(self):
        obs = {"wiki": {"top": "Dune"}}
        history = {"2025-12-31": day("2025-12-31", {"wiki": {"top": "Dune"}})}
        streak = predict.make_wiki(obs, history, "2026-01-01")
        fresh = predict.make_wiki(obs, {}, "2026-01-01")
        self.assertTrue(streak[0])
        self.assertFalse(fresh[0])
        self.assertGreater(streak[1], fresh[1])


class TestVerdictBlock(unittest.TestCase):
    """The failure list has to be derived, not written.

    It was prose once. Nine days in it had inverted — naming Wikipedia as the
    outright failure after Wikipedia had gone on to beat the baseline and
    Bitcoin had dropped below it. These tests exist so that cannot recur.
    """

    def _ledger(self, rows, graded=40):
        by_predictor = {}
        for pid, row in rows.items():
            row.setdefault("question", pid)
            row.setdefault("rule", "r")
            row.setdefault("family", pid)
            row.setdefault("variant", "rule")
            by_predictor[pid] = row
        return {
            "totals": {"graded": graded, "correct": 0, "days": 9},
            "baselines": {"coin_flip": {"brier": predict.COIN_FLIP_BRIER}},
            "by_predictor": by_predictor,
        }

    def test_names_the_question_below_the_coin_flip(self):
        led = self._ledger({
            "btc_up": {"graded": 8, "correct": 2, "accuracy": 0.25,
                       "brier": 0.269, "log_loss": 0.73,
                       "base_rate": 0.62, "majority_accuracy": 0.62,
                       "question": "Bitcoin question."},
            "wiki_hold": {"graded": 8, "correct": 5, "accuracy": 0.62,
                          "brier": 0.220, "log_loss": 0.63,
                          "base_rate": 0.75, "majority_accuracy": 0.75,
                          "question": "Wikipedia question."},
        })
        text = "\n".join(predict.verdict_block(led))
        self.assertIn("Bitcoin question.", text)
        self.assertIn("worse than shrugging", text)
        # The one that beats the baseline must not be listed as a failure.
        self.assertNotIn("Wikipedia question.", text)

    def test_flags_a_one_sided_question(self):
        led = self._ledger({
            "quake_m5": {"graded": 8, "correct": 8, "accuracy": 1.0,
                         "brier": 0.013, "log_loss": 0.11,
                         "base_rate": 1.0, "majority_accuracy": 1.0,
                         "question": "Quake question."},
        })
        text = "\n".join(predict.verdict_block(led))
        self.assertIn("Quake question.", text)
        self.assertIn("one-sided", text)
        self.assertIn("measures the question", text)

    def test_a_balanced_question_is_not_called_one_sided(self):
        """A 25% base rate is not one-sided; the old prose claimed it was 0%."""
        led = self._ledger({
            "kp_storm": {"graded": 8, "correct": 7, "accuracy": 0.88,
                         "brier": 0.124, "log_loss": 0.42,
                         "base_rate": 0.25, "majority_accuracy": 0.75,
                         "question": "Kp question."},
        })
        text = "\n".join(predict.verdict_block(led))
        self.assertNotIn("Kp question.", text)

    def test_reports_when_no_model_has_settled(self):
        led = self._ledger({
            "btc_up": {"graded": 8, "correct": 2, "accuracy": 0.25,
                       "brier": 0.269, "log_loss": 0.73,
                       "base_rate": 0.62, "majority_accuracy": 0.62},
            "btc_up_ml": {"graded": 0, "correct": 0, "accuracy": None,
                          "brier": None, "log_loss": None,
                          "base_rate": None, "majority_accuracy": None,
                          "variant": "model"},
        })
        self.assertIn("zero", "\n".join(predict.verdict_block(led)))

    def test_empty_ledger_does_not_crash(self):
        led = self._ledger({}, graded=0)
        text = "\n".join(predict.verdict_block(led))
        self.assertIn("Nothing has settled yet", text)

    def test_block_is_delimited_by_its_markers(self):
        led = self._ledger({})
        block = predict.verdict_block(led)
        self.assertEqual(block[0], "<!-- VERDICT:START -->")
        self.assertEqual(block[-1], "<!-- VERDICT:END -->")


class TestPerQuestionBaseRate(unittest.TestCase):
    def test_by_predictor_carries_its_own_base_rate(self):
        """Without it, a one-sided question cannot be told from real skill."""
        days = {
            "2026-01-01": day(
                "2026-01-01",
                {"quakes": {"max_mag": 6.1}},
                [bet("quake_m5", "2026-01-01", True, 0.85)],
            ),
            "2026-01-02": day("2026-01-02", {"quakes": {"max_mag": 6.4}}),
        }
        grades = predict.grade_all(days)
        led = predict.build_ledger(days, grades, "2026-01-02")
        row = led["by_predictor"]["quake_m5"]
        self.assertEqual(row["base_rate"], 1.0)
        self.assertEqual(row["majority_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

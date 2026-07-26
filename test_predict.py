#!/usr/bin/env python3
"""Offline tests for the grading half of Hindsight.

The fetching half is allowed to fail — a dead API just means no call that day.
The *grading* half is not: if it is wrong, the public track record is a lie.
So everything below runs on synthetic archives with no network access.
"""

import unittest

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
        self.assertEqual(ledger["totals"]["graded"], 0)


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
            result = spec["make"](obs, {}, "2026-01-01")
            self.assertIsNotNone(result, pid)
            _, confidence, _ = result
            self.assertGreaterEqual(confidence, 0.5, pid)
            self.assertLessEqual(confidence, 0.95, pid)

    def test_predictors_return_none_without_data(self):
        for pid, spec in predict.PREDICTORS.items():
            self.assertIsNone(spec["make"]({}, {}, "2026-01-01"), pid)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)

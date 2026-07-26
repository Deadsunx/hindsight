#!/usr/bin/env python3
"""Offline tests for the learning half of Hindsight.

The single most important property here is **no lookahead**. If a feature or a
training set can see past the day it is predicting, the model's record is
fraudulent no matter how good the git timestamps look. Those tests come first
and are the reason this file exists.
"""

import random
import unittest

import learn
import predict


def day(date, observations=None, predictions=None):
    return {
        "date": date,
        "fetched_at": f"{date}T04:00:00+00:00",
        "observations": observations or {},
        "predictions": predictions or [],
    }


def build_archive(n, start="2026-01-01", seed=3):
    """A synthetic archive with enough shape for features to bite."""
    rng = random.Random(seed)
    days, price, temp = {}, 60000.0, 30.0
    for i in range(n):
        date = predict.shift(start, i)
        price *= 1 + rng.gauss(0, 0.02)
        temp += rng.gauss(0, 1.5)
        days[date] = day(
            date,
            {
                "btc": {"usd": round(price, 2), "usd_24h_change": round(rng.gauss(0, 2), 2)},
                "kp": {"max_kp": round(rng.uniform(0.5, 6.0), 2)},
                "quakes": {"max_mag": round(rng.uniform(4.0, 7.0), 1), "count": rng.randint(80, 200)},
                "wiki": {"top": rng.choice(["Dune", "Cleopatra"]), "views": rng.randint(50000, 500000)},
                "hn": {"top_id": rng.randint(1, 10**7), "top10": [rng.randint(1, 10**7) for _ in range(10)]},
                "weather": {
                    "actual_prev": round(temp, 1),
                    "actual_prev_date": predict.shift(date, -1),
                    "today": round(temp + 1, 1),
                    "forecast_next": round(temp + rng.gauss(0, 2), 1),
                    "forecast_next_date": predict.shift(date, 1),
                },
            },
        )
    return days


class TestNoLookahead(unittest.TestCase):
    """Features must be blind to everything after the day they describe."""

    def test_features_ignore_future_days(self):
        full = build_archive(60)
        target = predict.shift("2026-01-01", 30)
        truncated = {d: v for d, v in full.items() if d <= target}

        for family, featfn in learn.FEATURES.items():
            self.assertEqual(
                featfn(full, target),
                featfn(truncated, target),
                f"{family} reads the future",
            )

    def test_features_unchanged_when_future_is_corrupted(self):
        """Stronger: rewriting later days must not move today's features."""
        full = build_archive(60)
        target = predict.shift("2026-01-01", 30)
        before = {f: fn(full, target) for f, fn in learn.FEATURES.items()}

        for date in sorted(full):
            if date > target:
                full[date]["observations"] = {
                    "btc": {"usd": 1.0, "usd_24h_change": 999.0},
                    "kp": {"max_kp": 99.0},
                    "quakes": {"max_mag": 9.9, "count": 1},
                    "wiki": {"top": "Corrupted", "views": 1},
                    "hn": {"top_id": 1, "top10": [1] * 10},
                    "weather": {"actual_prev": 99.0, "actual_prev_date": date,
                                "today": 99.0, "forecast_next": 99.0,
                                "forecast_next_date": date},
                }
        after = {f: fn(full, target) for f, fn in learn.FEATURES.items()}
        self.assertEqual(before, after)

    def test_training_set_excludes_the_day_being_predicted(self):
        days = build_archive(40)
        grades = [
            {"id": "btc_up", "made_on": predict.shift("2026-01-01", i),
             "actual": i % 2 == 0}
            for i in range(40)
        ]
        today = predict.shift("2026-01-01", 20)
        X, y = learn.training_set("btc_up", days, grades, today)
        self.assertEqual(len(X), len(y))
        # 20 candidate days before `today`, minus early ones lacking 2 prices.
        self.assertLessEqual(len(X), 20)
        self.assertGreater(len(X), 0)

    def test_training_set_ignores_other_families(self):
        days = build_archive(40)
        grades = [
            {"id": "kp_storm", "made_on": predict.shift("2026-01-01", i), "actual": True}
            for i in range(40)
        ]
        X, _ = learn.training_set("btc_up", days, grades, predict.shift("2026-01-01", 30))
        self.assertEqual(X, [])


class TestLogistic(unittest.TestCase):
    def test_learns_a_separable_signal(self):
        # y is 1 exactly when the first feature is positive.
        X = [[v, 0.3] for v in (-4, -3, -2, -1, 1, 2, 3, 4)]
        y = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        model = learn.Logistic().fit(X, y)
        self.assertGreater(model.predict_proba([3.5, 0.3]), 0.7)
        self.assertLess(model.predict_proba([-3.5, 0.3]), 0.3)

    def test_is_deterministic(self):
        X = [[v, v * 0.5] for v in range(-6, 7)]
        y = [0.0 if v < 0 else 1.0 for v in range(-6, 7)]
        a = learn.Logistic().fit(X, y).predict_proba([2.0, 1.0])
        b = learn.Logistic().fit(X, y).predict_proba([2.0, 1.0])
        self.assertEqual(a, b)

    def test_constant_feature_does_not_divide_by_zero(self):
        X = [[1.0, v] for v in (-2, -1, 1, 2)]
        y = [0.0, 0.0, 1.0, 1.0]
        model = learn.Logistic().fit(X, y)
        self.assertTrue(0.0 <= model.predict_proba([1.0, 2.0]) <= 1.0)

    def test_probabilities_stay_in_range(self):
        X = [[v] for v in range(-50, 50)]
        y = [0.0 if v < 0 else 1.0 for v in range(-50, 50)]
        model = learn.Logistic().fit(X, y)
        for probe in (-1000.0, 0.0, 1000.0):
            p = model.predict_proba([probe])
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_sigmoid_is_stable_at_extremes(self):
        self.assertAlmostEqual(learn.sigmoid(0.0), 0.5)
        self.assertGreater(learn.sigmoid(800.0), 0.99)
        self.assertLess(learn.sigmoid(-800.0), 0.01)


class TestColdStart(unittest.TestCase):
    def test_no_bet_below_the_training_minimum(self):
        days = build_archive(10)
        grades = [
            {"id": "btc_up", "made_on": predict.shift("2026-01-01", i), "actual": True}
            for i in range(5)
        ]
        self.assertIsNone(
            learn.probability("btc_up", days, grades, predict.shift("2026-01-01", 9))
        )

    def test_one_sided_history_falls_back_to_laplace(self):
        """Never claim certainty just because nothing has gone wrong yet."""
        days = build_archive(60)
        today = predict.shift("2026-01-01", 50)
        grades = [
            {"id": "hn_fade", "made_on": predict.shift("2026-01-01", i), "actual": True}
            for i in range(45)
        ]
        result = learn.probability("hn_fade", days, grades, today)
        self.assertIsNotNone(result)
        p, detail = result
        self.assertEqual(detail["fallback"], "laplace")
        self.assertLess(p, 1.0)
        self.assertGreater(p, 0.9)

    def test_confidence_is_bounded(self):
        for p in (0.0, 0.001, 0.5, 0.999, 1.0):
            call, confidence = learn.call_and_confidence(p)
            self.assertGreaterEqual(confidence, 0.5)
            self.assertLessEqual(confidence, learn.MAX_CONFIDENCE)
            self.assertEqual(call, p >= 0.5)

    def test_confidence_is_in_the_call_not_the_event(self):
        call, confidence = learn.call_and_confidence(0.2)
        self.assertFalse(call)
        self.assertAlmostEqual(confidence, 0.8)


class TestModelPredictors(unittest.TestCase):
    def test_model_variant_sits_out_until_trained(self):
        days = build_archive(5)
        obs = days[predict.shift("2026-01-01", 4)]["observations"]
        spec = predict.PREDICTORS["btc_up_ml"]
        self.assertIsNone(spec["make"](obs, days, predict.shift("2026-01-01", 4), []))

    def test_model_variant_bets_once_it_has_history(self):
        days = build_archive(60)
        today = predict.shift("2026-01-01", 50)
        grades = [
            {"id": "kp_storm", "made_on": predict.shift("2026-01-01", i),
             "actual": i % 3 == 0}
            for i in range(45)
        ]
        result = predict.PREDICTORS["kp_storm_ml"]["make"](
            days[today]["observations"], days, today, grades
        )
        self.assertIsNotNone(result)
        call, confidence, basis = result
        self.assertIsInstance(call, bool)
        self.assertGreaterEqual(confidence, 0.5)
        self.assertLessEqual(confidence, learn.MAX_CONFIDENCE)
        self.assertIn("model", basis)
        self.assertGreaterEqual(basis["model"]["n_train"], learn.MIN_TRAIN)

    def test_model_borrows_the_rules_basis_so_resolvers_still_work(self):
        """hn_fade resolves via basis['top_id']; the model must carry it."""
        days = build_archive(60)
        today = predict.shift("2026-01-01", 50)
        grades = [
            {"id": "hn_fade", "made_on": predict.shift("2026-01-01", i),
             "actual": i % 4 != 0}
            for i in range(45)
        ]
        result = predict.PREDICTORS["hn_fade_ml"]["make"](
            days[today]["observations"], days, today, grades
        )
        self.assertIsNotNone(result)
        _, _, basis = result
        self.assertEqual(basis["top_id"], days[today]["observations"]["hn"]["top_id"])


class TestHeadToHead(unittest.TestCase):
    def _grade(self, family, variant, date, correct, confidence=0.7):
        return {
            "id": family if variant == "rule" else f"{family}_ml",
            "family": family, "variant": variant, "made_on": date,
            "call": True, "confidence": confidence,
            "actual": correct, "correct": correct,
        }

    def test_only_shared_days_are_compared(self):
        grades = [
            # day 1: both bet
            self._grade("btc_up", "rule", "2026-01-01", True),
            self._grade("btc_up", "model", "2026-01-01", False),
            # day 2: rule only — the model was still cold
            self._grade("btc_up", "rule", "2026-01-02", True),
        ]
        h2h = predict.build_head_to_head(grades)["btc_up"]
        self.assertEqual(h2h["shared_days"], 1)
        self.assertEqual(h2h["rule_correct"], 1)
        self.assertEqual(h2h["model_correct"], 0)

    def test_brier_is_computed_on_the_same_days(self):
        grades = [
            self._grade("btc_up", "rule", "2026-01-01", True, 0.6),
            self._grade("btc_up", "model", "2026-01-01", True, 0.9),
        ]
        h2h = predict.build_head_to_head(grades)["btc_up"]
        # both called yes and were right: (0.6-1)^2=0.16 vs (0.9-1)^2=0.01
        self.assertAlmostEqual(h2h["rule_brier"], 0.16, places=4)
        self.assertAlmostEqual(h2h["model_brier"], 0.01, places=4)

    def test_legacy_grades_without_variants_are_skipped(self):
        """Predictions archived before variants existed must not crash it."""
        grades = [{"id": "btc_up", "made_on": "2026-01-01", "call": True,
                   "confidence": 0.6, "actual": True, "correct": True}]
        self.assertEqual(predict.build_head_to_head(grades), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)

#!/usr/bin/env python3
"""Tests for the auditor.

A verifier that cannot fail is decoration. Most of these tests are therefore
about detection: tamper with something, assert the audit notices.
"""

import unittest

import predict
import verify


class TestNormalise(unittest.TestCase):
    def test_drops_volatile_keys_at_any_depth(self):
        raw = {
            "updated": "2026-07-26T10:00:00Z",
            "totals": {"correct": 3},
            "by_predictor": {"btc_up": {"question": "anything", "graded": 4}},
        }
        self.assertEqual(
            verify.normalise(raw),
            {"totals": {"correct": 3}, "by_predictor": {"btc_up": {"graded": 4}}},
        )

    def test_drops_volatile_keys_inside_lists(self):
        raw = {"grades": [{"question": "q", "correct": True}]}
        self.assertEqual(verify.normalise(raw), {"grades": [{"correct": True}]})

    def test_a_timestamp_difference_alone_is_not_a_mismatch(self):
        a = {"updated": "2026-07-26T10:00:00Z", "totals": {"correct": 3}}
        b = {"updated": "2026-07-27T22:31:04Z", "totals": {"correct": 3}}
        self.assertEqual(verify.normalise(a), verify.normalise(b))


class TestDifferences(unittest.TestCase):
    def test_identical_structures_report_nothing(self):
        payload = {"totals": {"correct": 3, "brier": 0.21}, "grades": [1, 2, 3]}
        self.assertEqual(verify.differences(payload, dict(payload)), [])

    def test_reports_a_changed_number_with_its_path(self):
        diffs = verify.differences({"totals": {"correct": 3}}, {"totals": {"correct": 9}})
        self.assertEqual(len(diffs), 1)
        self.assertIn("totals.correct", diffs[0])
        self.assertIn("3", diffs[0])
        self.assertIn("9", diffs[0])

    def test_reports_missing_and_extra_keys(self):
        diffs = verify.differences({"a": 1, "b": 2}, {"a": 1, "c": 3})
        joined = " ".join(diffs)
        self.assertIn(".b", joined)
        self.assertIn(".c", joined)

    def test_reports_length_mismatch_on_lists(self):
        diffs = verify.differences({"g": [1, 2, 3]}, {"g": [1, 2]})
        self.assertTrue(any("length" in d for d in diffs))

    def test_output_is_capped(self):
        big_a = {str(i): i for i in range(50)}
        big_b = {str(i): i + 1 for i in range(50)}
        self.assertLessEqual(len(verify.differences(big_a, big_b, limit=8)), 8)

    def test_type_change_is_caught(self):
        """An accuracy silently switching null -> number must not slip past."""
        diffs = verify.differences({"accuracy": None}, {"accuracy": 0.99})
        self.assertEqual(len(diffs), 1)
        self.assertIn("NoneType", diffs[0])


class TestTamperDetection(unittest.TestCase):
    """End-to-end: build an archive, recompute, then corrupt it."""

    def _archive(self):
        return {
            "2026-01-01": {
                "date": "2026-01-01",
                "observations": {"btc": {"usd": 60000}, "quakes": {"max_mag": 6.1}},
                "predictions": [
                    {"id": "btc_up", "question": "q", "rule": "momentum",
                     "family": "btc_up", "variant": "rule",
                     "call": True, "confidence": 0.55, "basis": {},
                     "made_on": "2026-01-01", "resolve_on": "2026-01-02"},
                ],
            },
            "2026-01-02": {
                "date": "2026-01-02",
                "observations": {"btc": {"usd": 61000}, "quakes": {"max_mag": 4.2}},
                "predictions": [],
            },
        }

    def _ledger(self, days):
        grades = predict.grade_all(days)
        return predict.build_ledger(days, grades, max(days))

    def test_an_untouched_ledger_verifies(self):
        days = self._archive()
        self.assertEqual(
            verify.differences(
                verify.normalise(self._ledger(days)),
                verify.normalise(self._ledger(days)),
            ),
            [],
        )

    def test_inflating_the_correct_count_is_caught(self):
        days = self._archive()
        honest = self._ledger(days)
        cooked = self._ledger(days)
        cooked["totals"]["correct"] += 5
        diffs = verify.differences(verify.normalise(honest), verify.normalise(cooked))
        self.assertTrue(any("totals.correct" in d for d in diffs), diffs)

    def test_deleting_a_losing_grade_is_caught(self):
        """Quietly dropping a miss changes the recomputation, so it shows."""
        days = self._archive()
        honest = self._ledger(days)
        cooked = self._ledger(days)
        cooked["grades"] = []
        diffs = verify.differences(verify.normalise(honest), verify.normalise(cooked))
        self.assertTrue(diffs)

    def test_flipping_an_outcome_is_caught(self):
        days = self._archive()
        honest = self._ledger(days)
        cooked = self._ledger(days)
        cooked["grades"][0]["correct"] = not cooked["grades"][0]["correct"]
        diffs = verify.differences(verify.normalise(honest), verify.normalise(cooked))
        self.assertTrue(any("correct" in d for d in diffs), diffs)

    def test_rewriting_an_observation_changes_the_recomputation(self):
        """The deepest guarantee: the ledger is downstream of the archive.

        Editing yesterday's price to turn a loss into a win cannot be done
        quietly, because the recomputed ledger no longer matches the published
        one — and the archive edit itself lands in git history.
        """
        honest = self._ledger(self._archive())
        days = self._archive()
        days["2026-01-02"]["observations"]["btc"]["usd"] = 50000  # BTC "fell"
        self.assertTrue(
            verify.differences(verify.normalise(honest), verify.normalise(self._ledger(days)))
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

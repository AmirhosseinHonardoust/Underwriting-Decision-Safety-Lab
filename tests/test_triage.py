from __future__ import annotations

import unittest

import numpy as np

from underwriting.triage import APPLICANT_COLUMNS, applicant_to_frame, triage_decision


class TriageDecisionTests(unittest.TestCase):
    def test_auto_decide_when_confident(self) -> None:
        d = triage_decision(0.9, threshold=0.7)
        self.assertAlmostEqual(d.confidence, 0.9)
        self.assertTrue(d.auto_decide)
        self.assertEqual(d.decision, "AUTO-DECIDE")

    def test_review_when_unsure(self) -> None:
        d = triage_decision(0.55, threshold=0.7)
        self.assertAlmostEqual(d.confidence, 0.55)
        self.assertFalse(d.auto_decide)
        self.assertEqual(d.decision, "REVIEW")

    def test_threshold_boundary_is_inclusive(self) -> None:
        # confidence exactly equal to the threshold auto-decides (>=).
        d = triage_decision(0.8, threshold=0.8)
        self.assertTrue(d.auto_decide)

    def test_rejection_confidence_is_symmetric(self) -> None:
        # p and 1 - p must yield identical confidence.
        self.assertAlmostEqual(
            triage_decision(0.2, 0.7).confidence,
            triage_decision(0.8, 0.7).confidence,
        )

    def test_invalid_probability_raises(self) -> None:
        for bad in (-0.01, 1.5):
            with self.assertRaises(ValueError):
                triage_decision(bad, threshold=0.7)

    def test_properties_over_random_probabilities(self) -> None:
        rng = np.random.default_rng(0)
        for p in rng.random(500):
            thr = float(rng.uniform(0.5, 0.99))
            d = triage_decision(float(p), thr)
            self.assertGreaterEqual(d.confidence, 0.5)
            self.assertLessEqual(d.confidence, 1.0)
            self.assertEqual(d.auto_decide, d.confidence >= thr)
            self.assertEqual(d.decision, "AUTO-DECIDE" if d.auto_decide else "REVIEW")


class ApplicantFrameTests(unittest.TestCase):
    def test_builds_single_row_with_expected_columns(self) -> None:
        df = applicant_to_frame(
            age=35,
            gender="Female",
            marital_status="Married",
            annual_income=80000,
            loan_amount=25000,
            credit_score=700,
            num_dependents=1,
            existing_loans_count=1,
            employment_status="Employed",
        )
        self.assertEqual(len(df), 1)
        self.assertEqual(list(df.columns), list(APPLICANT_COLUMNS))
        self.assertEqual(df.iloc[0]["credit_score"], 700)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from underwriting.slices import (
    add_numeric_slice_bins,
    build_slice_report,
    save_slice_artifacts,
    slice_metrics_for_group,
    summarize_slice_report,
)


class SliceReportingTests(unittest.TestCase):
    def _predictions(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "gender": ["Female", "Male", "Female", "Male", "Female", "Male"],
                "marital_status": ["Single", "Married", "Single", "Married", "Divorced", "Single"],
                "employment_status": [
                    "Employed",
                    "Employed",
                    "Unemployed",
                    "Self-employed",
                    "Employed",
                    "Unemployed",
                ],
                "age": [22, 45, 35, 60, 29, 52],
                "credit_score": [620, 740, 680, 810, 590, 710],
                "annual_income": [45000, 90000, 62000, 120000, 38000, 76000],
                "y_true": [0, 1, 1, 1, 0, 1],
                "p_approve": [0.30, 0.88, 0.65, 0.92, 0.40, 0.78],
                "p_reject": [0.70, 0.12, 0.35, 0.08, 0.60, 0.22],
                "pred_label": [0, 1, 1, 1, 0, 1],
                "confidence": [0.70, 0.88, 0.65, 0.92, 0.60, 0.78],
                "auto_decide": [True, True, False, True, False, True],
            }
        )

    def test_add_numeric_slice_bins(self) -> None:
        df = self._predictions()
        out = add_numeric_slice_bins(df)

        self.assertIn("age_band", out.columns)
        self.assertIn("credit_score_band", out.columns)
        self.assertIn("annual_income_band", out.columns)

    def test_slice_metrics_for_group(self) -> None:
        group = self._predictions()
        metrics = slice_metrics_for_group(group, min_count=2)

        self.assertEqual(metrics["n"], len(group))
        self.assertAlmostEqual(metrics["auto_decision_rate"], 4 / 6)
        self.assertAlmostEqual(metrics["review_rate"], 2 / 6)
        self.assertEqual(metrics["error_rate"], 0.0)
        self.assertFalse(metrics["is_small_slice"])

    def test_build_slice_report_has_expected_columns(self) -> None:
        report = build_slice_report(self._predictions(), min_count=2)

        required = {
            "slice_feature",
            "slice_value",
            "n",
            "observed_approval_rate",
            "mean_predicted_approval_probability",
            "auto_decision_rate",
            "review_rate",
            "auto_accuracy",
            "error_rate",
            "false_approval_rate",
            "false_rejection_rate",
            "ece",
            "is_small_slice",
        }
        self.assertTrue(required.issubset(report.columns))
        self.assertGreater(len(report), 0)

    def test_build_slice_report_rejects_missing_required_columns(self) -> None:
        df = self._predictions().drop(columns=["p_approve"])

        with self.assertRaisesRegex(ValueError, "missing required columns"):
            build_slice_report(df)

    def test_summary_and_artifacts(self) -> None:
        report = build_slice_report(self._predictions(), min_count=2)
        summary = summarize_slice_report(report)

        self.assertIn("n_slices", summary)
        self.assertIn("max_auto_decision_rate_gap", summary)

        with tempfile.TemporaryDirectory() as tmp:
            paths = save_slice_artifacts(report, tmp)

            self.assertTrue(paths["csv"].exists())
            self.assertTrue(paths["json"].exists())
            self.assertTrue(paths["summary"].exists())

            loaded_summary = json.loads(Path(paths["summary"]).read_text())
            self.assertEqual(loaded_summary["n_slices"], summary["n_slices"])


if __name__ == "__main__":
    unittest.main()

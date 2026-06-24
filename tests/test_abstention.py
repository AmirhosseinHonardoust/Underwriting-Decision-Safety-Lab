from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.abstention import coverage_curve, recommend_threshold
from src.data import load_csv


class CoverageCurveTests(unittest.TestCase):
    def test_all_negative_subset_yields_zero_f1_without_warning(self) -> None:
        # All labels and predictions are the negative class: f1 is ill-defined and
        # must resolve to 0.0 via zero_division=0, with no warning raised.
        y_true = np.zeros(20, dtype=int)
        p_approve = np.full(20, 0.1)  # confidence 0.9, predicted class 0
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            curve = coverage_curve(y_true, p_approve, np.array([0.5]))
        row = curve.iloc[0]
        self.assertEqual(row["coverage"], 1.0)
        self.assertEqual(row["f1"], 0.0)

    def test_empty_auto_subset_is_nan(self) -> None:
        y_true = np.array([0, 1, 0, 1])
        p_approve = np.array([0.55, 0.45, 0.6, 0.4])  # max confidence 0.6
        curve = coverage_curve(y_true, p_approve, np.array([0.95]))
        row = curve.iloc[0]
        self.assertEqual(row["coverage"], 0.0)
        self.assertTrue(np.isnan(row["accuracy"]))
        self.assertTrue(np.isnan(row["f1"]))

    def test_recommend_threshold_returns_expected_keys(self) -> None:
        y_true = np.array([0, 1, 0, 1, 1, 0])
        p_approve = np.array([0.2, 0.8, 0.3, 0.7, 0.9, 0.1])
        curve = coverage_curve(y_true, p_approve, np.linspace(0.5, 0.99, 10))
        policy = recommend_threshold(curve, target_coverage=0.5)
        self.assertEqual(
            set(policy),
            {
                "recommended_threshold",
                "expected_coverage",
                "expected_accuracy_auto",
                "expected_f1_auto",
            },
        )


class LoadCsvTests(unittest.TestCase):
    def test_load_csv_reads_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.csv"
            path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
            df = load_csv(str(path))
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(list(df.columns), ["a", "b"])
        self.assertEqual(len(df), 2)


if __name__ == "__main__":
    unittest.main()

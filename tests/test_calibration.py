from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from underwriting.calibration import calibration_bins, expected_calibration_error
from underwriting.plots import plot_reliability_diagram


class CalibrationTests(unittest.TestCase):
    def test_ece_uses_observed_positive_rate_not_accuracy(self) -> None:
        y_true = np.array([0, 0])
        p_approve = np.array([0.2, 0.2])

        ece = expected_calibration_error(y_true, p_approve, n_bins=5)

        # Correct binary calibration: observed approval rate is 0.0, mean predicted is 0.2.
        self.assertAlmostEqual(ece, 0.2, places=7)

    def test_perfect_binary_probabilities_have_zero_ece(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        p_approve = np.array([0.0, 0.0, 1.0, 1.0])

        ece = expected_calibration_error(y_true, p_approve, n_bins=10)

        self.assertAlmostEqual(ece, 0.0, places=7)

    def test_calibration_bins_report_observed_positive_rate(self) -> None:
        y_true = np.array([0, 1, 1])
        p_approve = np.array([0.2, 0.2, 0.2])

        rows = calibration_bins(y_true, p_approve, n_bins=5)
        non_empty = [row for row in rows if row["count"] > 0]

        self.assertEqual(len(non_empty), 1)
        self.assertEqual(non_empty[0]["count"], 3)
        self.assertAlmostEqual(non_empty[0]["mean_predicted_probability"], 0.2)
        self.assertAlmostEqual(non_empty[0]["observed_positive_rate"], 2 / 3)

    def test_invalid_probability_values_raise(self) -> None:
        with self.assertRaises(ValueError):
            expected_calibration_error(np.array([0, 1]), np.array([0.2, 1.2]))

    def test_reliability_diagram_writes_file(self) -> None:
        y_true = np.array([0, 0, 1, 1, 1])
        p_approve = np.array([0.1, 0.2, 0.7, 0.8, 0.9])

        with tempfile.TemporaryDirectory() as tmp:
            outpath = Path(tmp) / "reliability.png"
            plot_reliability_diagram(y_true, p_approve, outpath, n_bins=5)
            self.assertTrue(outpath.exists())
            self.assertGreater(outpath.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

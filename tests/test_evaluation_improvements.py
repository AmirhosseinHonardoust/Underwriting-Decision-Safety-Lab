from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import (
    classification_metrics_at_threshold,
    compute_baseline_metrics,
    probability_metrics,
    select_policy_variants,
)
from src.pipeline import run

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "loanapproval.csv"


class EvaluationImprovementTests(unittest.TestCase):
    def test_probability_metrics_include_average_precision_and_brier(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.25, 0.7, 0.9])

        metrics = probability_metrics(y_true, p)

        self.assertIn("average_precision", metrics)
        self.assertIn("brier", metrics)
        self.assertIn("roc_auc", metrics)
        self.assertGreaterEqual(metrics["average_precision"], 0.0)
        self.assertLessEqual(metrics["average_precision"], 1.0)
        self.assertGreaterEqual(metrics["brier"], 0.0)

    def test_classification_metrics_at_threshold_are_bounded(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        p = np.array([0.2, 0.6, 0.7, 0.9])

        metrics = classification_metrics_at_threshold(y_true, p, threshold=0.5)

        self.assertEqual(metrics["threshold"], 0.5)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)
        self.assertGreaterEqual(metrics["f1"], 0.0)
        self.assertLessEqual(metrics["f1"], 1.0)

    def test_baseline_metrics_have_expected_strategies(self) -> None:
        X_train = pd.DataFrame({"x": [0, 1, 2, 3, 4, 5]})
        y_train = np.array([0, 0, 0, 1, 1, 1])
        X_test = pd.DataFrame({"x": [6, 7, 8, 9]})
        y_test = np.array([0, 1, 0, 1])

        baselines = compute_baseline_metrics(X_train, y_train, X_test, y_test)

        self.assertEqual(set(baselines), {"majority_class", "empirical_prior", "stratified_random"})
        for metrics in baselines.values():
            self.assertIn("average_precision", metrics)
            self.assertIn("brier", metrics)
            self.assertIn("strategy", metrics)

    def test_policy_variants_return_named_policy_options(self) -> None:
        curve = pd.DataFrame(
            {
                "threshold": [0.50, 0.70, 0.90],
                "coverage": [1.00, 0.70, 0.30],
                "accuracy": [0.80, 0.90, 0.98],
                "f1": [0.82, 0.91, 0.97],
            }
        )

        variants = select_policy_variants(curve, target_coverage=0.70)

        expected = {
            "target_coverage",
            "quality_first",
            "high_coverage",
            "balanced",
            "conservative_review",
        }
        self.assertEqual(set(variants), expected)
        self.assertAlmostEqual(variants["target_coverage"]["coverage"], 0.70)
        for policy in variants.values():
            self.assertIn("description", policy)
            self.assertIn("threshold", policy)
            self.assertIn("coverage", policy)
            self.assertTrue(0.0 <= policy["coverage"] <= 1.0)

    def test_pipeline_writes_enhanced_evaluation_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out_dir = tmp / "outputs"
            fig_dir = tmp / "figures"

            run(
                input_path=str(DATA_PATH),
                out_dir=str(out_dir),
                figures_dir=str(fig_dir),
                recommend_target_coverage=0.70,
                random_state=42,
            )

            expected_outputs = [
                out_dir / "baseline_metrics.json",
                out_dir / "policy_variants.json",
                out_dir / "evaluation_summary.json",
                fig_dir / "precision_recall_curve.png",
            ]

            for path in expected_outputs:
                with self.subTest(path=path.name):
                    self.assertTrue(path.exists(), f"Missing output: {path}")
                    self.assertGreater(path.stat().st_size, 0)

            summary = json.loads((out_dir / "evaluation_summary.json").read_text(encoding="utf-8"))
            self.assertIn("model_metrics", summary)
            self.assertIn("baseline_metrics", summary)
            self.assertIn("policy_variants", summary)
            self.assertIn("average_precision", summary["model_metrics"])


if __name__ == "__main__":
    unittest.main()

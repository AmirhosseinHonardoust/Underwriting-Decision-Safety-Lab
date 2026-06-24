from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from underwriting.abstention import coverage_curve, recommend_threshold
from underwriting.data import basic_quality_report, infer_spec
from underwriting.modeling import make_base_model, make_preprocessor, train_test_split_data
from underwriting.pipeline import run

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "loanapproval.csv"


class PipelineContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.df = pd.read_csv(DATA_PATH)

    def test_infer_spec_detects_target_id_and_feature_types(self) -> None:
        spec = infer_spec(self.df)

        self.assertEqual(spec.target, "loan_approved")
        self.assertIn("applicant_id", spec.id_cols)
        self.assertIn("age", spec.numeric_cols)
        self.assertIn("annual_income", spec.numeric_cols)
        self.assertIn("gender", spec.categorical_cols)
        self.assertIn("employment_status", spec.categorical_cols)

    def test_train_test_split_preserves_binary_target_rate(self) -> None:
        spec = infer_spec(self.df)
        df_model = self.df.drop(columns=spec.id_cols)
        split = train_test_split_data(df_model, spec.target, test_size=0.25, random_state=42)

        overall_rate = float(df_model[spec.target].mean())
        train_rate = float(np.mean(split.y_train))
        test_rate = float(np.mean(split.y_test))

        self.assertAlmostEqual(train_rate, overall_rate, delta=0.03)
        self.assertAlmostEqual(test_rate, overall_rate, delta=0.03)
        self.assertEqual(len(split.X_train) + len(split.X_test), len(df_model))

    def test_model_pipeline_can_fit_and_predict_probabilities(self) -> None:
        spec = infer_spec(self.df)
        sample = self.df.drop(columns=spec.id_cols).sample(n=160, random_state=7)
        split = train_test_split_data(sample, spec.target, test_size=0.25, random_state=7)

        preprocessor = make_preprocessor(spec.numeric_cols, spec.categorical_cols)
        model = make_base_model(random_state=7)
        pipeline = Pipeline([("pre", preprocessor), ("clf", model)])
        pipeline.fit(split.X_train, split.y_train)

        proba = pipeline.predict_proba(split.X_test)[:, 1]
        self.assertEqual(len(proba), len(split.X_test))
        self.assertTrue(np.all((proba >= 0.0) & (proba <= 1.0)))

    def test_coverage_curve_and_policy_contract(self) -> None:
        y_true = np.array([0, 0, 1, 1, 1, 0])
        p_approve = np.array([0.05, 0.20, 0.65, 0.85, 0.95, 0.45])
        thresholds = np.array([0.50, 0.70, 0.90])

        curve = coverage_curve(y_true, p_approve, thresholds)
        policy = recommend_threshold(curve, target_coverage=0.70)

        self.assertEqual(list(curve.columns), ["threshold", "coverage", "accuracy", "f1"])
        self.assertTrue(curve["coverage"].between(0, 1).all())
        self.assertIn("recommended_threshold", policy)
        self.assertIn("expected_coverage", policy)
        self.assertTrue(0.0 <= policy["expected_coverage"] <= 1.0)

    def test_basic_quality_report_contains_expected_fields(self) -> None:
        spec = infer_spec(self.df)
        report = basic_quality_report(self.df, spec)

        self.assertEqual(report["rows"], len(self.df))
        self.assertEqual(report["target"], "loan_approved")
        self.assertIn("missing_rate_by_col", report)
        self.assertIn("n_unique_by_col", report)
        self.assertIn("plausibility_hints", report)

    def test_pipeline_smoke_run_writes_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out_dir = tmp / "outputs"
            fig_dir = tmp / "figures"

            result = run(
                input_path=str(DATA_PATH),
                out_dir=str(out_dir),
                figures_dir=str(fig_dir),
                recommend_target_coverage=0.70,
                random_state=42,
            )

            expected_outputs = [
                out_dir / "metrics_overall.json",
                out_dir / "coverage_curve.csv",
                out_dir / "abstention_policy.json",
                out_dir / "test_predictions.csv",
                out_dir / "model.joblib",
                out_dir / "policy_card.md",
                out_dir / "data_quality.json",
                fig_dir / "confusion_matrix.png",
                fig_dir / "reliability_diagram.png",
                fig_dir / "probability_histograms.png",
                fig_dir / "coverage_vs_performance.png",
            ]

            for path in expected_outputs:
                with self.subTest(path=path.name):
                    self.assertTrue(path.exists(), f"Missing output: {path}")
                    self.assertGreater(path.stat().st_size, 0)

            self.assertIn("metrics", result)
            self.assertIn("policy", result)
            self.assertIn("recommended_threshold", result["policy"])


if __name__ == "__main__":
    unittest.main()

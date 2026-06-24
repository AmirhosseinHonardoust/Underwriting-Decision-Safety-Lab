from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from underwriting.pipeline import build_run_manifest, run
from underwriting.synthetic import generate_synthetic
from underwriting.validation import validate_underwriting_dataframe


class SyntheticDataTests(unittest.TestCase):
    EXPECTED_COLUMNS = [
        "applicant_id",
        "age",
        "gender",
        "marital_status",
        "annual_income",
        "loan_amount",
        "credit_score",
        "num_dependents",
        "existing_loans_count",
        "employment_status",
        "loan_approved",
    ]

    def test_schema_and_determinism(self) -> None:
        a = generate_synthetic(n_rows=300, random_state=7)
        b = generate_synthetic(n_rows=300, random_state=7)
        self.assertEqual(list(a.columns), self.EXPECTED_COLUMNS)
        self.assertEqual(len(a), 300)
        self.assertTrue(a.equals(b))  # same seed -> identical frame

    def test_passes_validation_and_has_both_classes(self) -> None:
        df = generate_synthetic(n_rows=400, random_state=1)
        self.assertEqual(set(df["loan_approved"].unique()), {0, 1})
        # Should not raise.
        validate_underwriting_dataframe(
            df,
            target="loan_approved",
            numeric_cols=[
                "age",
                "annual_income",
                "loan_amount",
                "credit_score",
                "num_dependents",
                "existing_loans_count",
            ],
            categorical_cols=["gender", "marital_status", "employment_status"],
        )

    def test_pipeline_runs_on_synthetic_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            csv = tmp_p / "synth.csv"
            generate_synthetic(n_rows=400, random_state=2).to_csv(csv, index=False)
            res = run(
                input_path=str(csv),
                out_dir=str(tmp_p / "out"),
                figures_dir=str(tmp_p / "fig"),
                recommend_target_coverage=0.70,
            )
            self.assertIn("metrics", res)
            self.assertTrue((tmp_p / "out" / "run_manifest.json").exists())


class RunManifestTests(unittest.TestCase):
    def test_manifest_fields_and_input_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / "synth.csv"
            df = generate_synthetic(n_rows=50, random_state=3)
            df.to_csv(csv, index=False)
            manifest = build_run_manifest(
                df,
                input_path=str(csv),
                random_state=42,
                calibration_method="sigmoid",
                target_coverage=0.7,
            )
            for key in (
                "generated_at",
                "package_version",
                "python_version",
                "input_sha256",
                "input_rows",
                "dependency_versions",
            ):
                self.assertIn(key, manifest)
            self.assertEqual(manifest["input_rows"], 50)
            self.assertEqual(len(manifest["input_sha256"]), 64)  # sha256 hex digest


if __name__ == "__main__":
    unittest.main()

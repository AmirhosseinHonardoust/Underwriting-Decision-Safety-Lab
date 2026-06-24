from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from underwriting.model_card import build_model_card_html
from underwriting.pipeline import run
from underwriting.synthetic import generate_synthetic


class ModelCardUnitTests(unittest.TestCase):
    def test_builds_self_contained_html(self) -> None:
        html = build_model_card_html(
            metrics={"accuracy": 0.9, "f1": 0.88, "roc_auc": 0.95, "ece": 0.05},
            policy={"recommended_threshold": 0.85, "expected_coverage": 0.7},
            baseline_metrics={"majority_class": {"accuracy": 0.7, "f1": 0.8, "brier": 0.2}},
            slice_summary={"n_slices": 20, "max_error_rate_gap": 0.14},
            data_quality={"rows": 1000, "cols": 11},
            figures_dir="/nonexistent",  # missing figures are skipped gracefully
        )
        self.assertIn("<!doctype html>", html.lower())
        self.assertIn("Headline metrics", html)
        self.assertIn("Recommended decision policy", html)
        self.assertIn("0.900", html)  # formatted accuracy

    def test_handles_missing_and_nan_values(self) -> None:
        html = build_model_card_html(
            metrics={"accuracy": float("nan"), "roc_auc": None},
            policy={},
            baseline_metrics={},
            slice_summary={},
            data_quality={},
            figures_dir="/nonexistent",
        )
        self.assertIn("n/a", html)


class ModelCardPipelineTests(unittest.TestCase):
    def test_pipeline_writes_self_contained_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            csv = tmp_p / "synth.csv"
            generate_synthetic(n_rows=400, random_state=5).to_csv(csv, index=False)
            run(
                input_path=str(csv),
                out_dir=str(tmp_p / "out"),
                figures_dir=str(tmp_p / "fig"),
                recommend_target_coverage=0.70,
            )
            card = tmp_p / "out" / "model_card.html"
            self.assertTrue(card.exists())
            text = card.read_text(encoding="utf-8")
            # Figures should be embedded as base64 data URIs (fully portable).
            self.assertIn("data:image/png;base64,", text)
            self.assertGreater(len(text), 10_000)


if __name__ == "__main__":
    unittest.main()

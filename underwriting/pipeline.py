from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .abstention import coverage_curve, recommend_threshold
from .calibration import calibrate, expected_calibration_error
from .data import basic_quality_report, infer_spec, load_csv
from .evaluation import compute_baseline_metrics, select_policy_variants
from .modeling import (
    compute_binary_metrics,
    make_base_model,
    make_preprocessor,
    train_test_split_data,
)
from .plots import (
    plot_confusion_matrix,
    plot_coverage_vs_performance,
    plot_precision_recall_curve,
    plot_probability_histograms,
    plot_reliability_diagram,
    plot_slice_error_rates,
    plot_slice_review_rates,
)
from .slices import build_slice_report, save_slice_artifacts, summarize_slice_report
from .validation import validate_underwriting_dataframe


def _package_version() -> str:
    try:
        return importlib_metadata.version("underwriting-decision-safety-lab")
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for dist in ("numpy", "pandas", "scikit-learn", "joblib"):
        try:
            versions[dist] = importlib_metadata.version(dist)
        except importlib_metadata.PackageNotFoundError:
            versions[dist] = "unknown"
    return versions


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_manifest(
    df: pd.DataFrame,
    *,
    input_path: str,
    random_state: int,
    calibration_method: str,
    target_coverage: float,
) -> dict[str, Any]:
    """Capture run provenance: input fingerprint, config, and environment."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "package_version": _package_version(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "input_path": str(input_path),
        "input_sha256": _file_sha256(input_path),
        "input_rows": int(len(df)),
        "input_columns": list(df.columns),
        "random_state": int(random_state),
        "calibration_method": str(calibration_method),
        "target_coverage": float(target_coverage),
        "dependency_versions": _dependency_versions(),
    }


def run(
    input_path: str,
    out_dir: str = "outputs",
    figures_dir: str = "reports/figures",
    calibration_method: str = "sigmoid",
    recommend_target_coverage: float = 0.70,
    random_state: int = 42,
) -> dict:
    out_dir_p = Path(out_dir)
    fig_dir_p = Path(figures_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    fig_dir_p.mkdir(parents=True, exist_ok=True)

    df = load_csv(input_path)
    spec = infer_spec(df)
    validate_underwriting_dataframe(
        df,
        target=spec.target,
        numeric_cols=spec.numeric_cols,
        categorical_cols=spec.categorical_cols,
    )

    df_model = df.drop(columns=spec.id_cols) if spec.id_cols else df.copy()
    split = train_test_split_data(df_model, spec.target, test_size=0.25, random_state=random_state)

    pre = make_preprocessor(spec.numeric_cols, spec.categorical_cols)
    base = make_base_model(random_state=random_state)

    base_pipe = Pipeline([("pre", pre), ("clf", base)])
    cal = calibrate(base_pipe, split.X_train, split.y_train, method=calibration_method, cv=3)

    p_test = cal.predict_proba(split.X_test)[:, 1]
    y_pred = (p_test >= 0.5).astype(int)

    metrics: dict[str, Any] = dict(compute_binary_metrics(split.y_test, p_test, y_pred))
    metrics["ece"] = float(expected_calibration_error(split.y_test, p_test, n_bins=10))
    metrics["labels"] = ["reject", "approve"]
    metrics["calibration_method"] = str(calibration_method)

    baseline_metrics = compute_baseline_metrics(
        split.X_train, split.y_train, split.X_test, split.y_test
    )

    thresholds = np.linspace(0.50, 0.99, 40)
    curve = coverage_curve(split.y_test, p_test, thresholds)
    policy: dict[str, Any] = dict(
        recommend_threshold(curve, target_coverage=float(recommend_target_coverage))
    )
    policy["target_coverage"] = float(recommend_target_coverage)
    policy["calibration_method"] = str(calibration_method)
    policy_variants = select_policy_variants(
        curve, target_coverage=float(recommend_target_coverage)
    )

    (out_dir_p / "metrics_overall.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir_p / "baseline_metrics.json").write_text(
        json.dumps(baseline_metrics, indent=2), encoding="utf-8"
    )
    (out_dir_p / "policy_variants.json").write_text(
        json.dumps(policy_variants, indent=2), encoding="utf-8"
    )
    curve.to_csv(out_dir_p / "coverage_curve.csv", index=False)
    (out_dir_p / "abstention_policy.json").write_text(
        json.dumps(policy, indent=2), encoding="utf-8"
    )

    preds = split.X_test.copy()
    preds["y_true"] = split.y_test
    preds["p_approve"] = p_test
    preds["p_reject"] = 1.0 - p_test
    preds["pred_label"] = (p_test >= 0.5).astype(int)
    preds["confidence"] = np.maximum(p_test, 1.0 - p_test)
    preds["auto_decide"] = preds["confidence"] >= float(policy["recommended_threshold"])
    preds.to_csv(out_dir_p / "test_predictions.csv", index=False)

    slice_report = build_slice_report(preds)
    slice_paths = save_slice_artifacts(slice_report, out_dir_p)
    slice_summary = summarize_slice_report(slice_report)

    evaluation_summary = {
        "model_metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "recommended_policy": policy,
        "policy_variants": policy_variants,
        "slice_summary": slice_summary,
        "slice_artifacts": {key: str(path) for key, path in slice_paths.items()},
    }
    (out_dir_p / "evaluation_summary.json").write_text(
        json.dumps(evaluation_summary, indent=2), encoding="utf-8"
    )

    joblib.dump({"model": cal, "spec": spec.__dict__}, out_dir_p / "model.joblib")

    decision_line = (
        "Use a calibrated model to approve/reject automatically when confident; "
        "otherwise route to manual review."
    )
    card = f"""# Policy Card — Underwriting Decision Safety

## Decision
{decision_line}

## Rule
- Compute p(approve)
- confidence = max(p(approve), 1 - p(approve))
- If confidence >= {policy['recommended_threshold']:.2f}: AUTO-DECIDE
- Else: REVIEW

## Target
- Target auto-decision coverage: {policy['target_coverage']:.2f}
- Expected coverage at recommended threshold: {policy['expected_coverage']:.2f}

## Expected auto-decision performance (test)
- Accuracy (auto-decisions): {policy['expected_accuracy_auto']:.3f}
- F1 (auto-decisions): {policy['expected_f1_auto']:.3f}

## Calibration and ranking health (test)
- ECE: {metrics['ece']:.4f}
- Brier: {metrics['brier']:.4f}
- Average precision: {metrics['average_precision']:.4f}
 
## Monitoring triggers (suggested)
- If coverage shifts by > 10% vs baseline, re-check score distribution.
- If ECE doubles vs baseline, re-calibrate or retrain.
- If approval rate shifts sharply, check for population/mix shift.
"""
    (out_dir_p / "policy_card.md").write_text(card, encoding="utf-8")

    plot_confusion_matrix(split.y_test, y_pred, fig_dir_p / "confusion_matrix.png")
    plot_reliability_diagram(split.y_test, p_test, fig_dir_p / "reliability_diagram.png")
    plot_probability_histograms(split.y_test, p_test, fig_dir_p / "probability_histograms.png")
    plot_precision_recall_curve(split.y_test, p_test, fig_dir_p / "precision_recall_curve.png")
    plot_coverage_vs_performance(curve, fig_dir_p / "coverage_vs_performance.png")
    plot_slice_review_rates(slice_report, fig_dir_p / "slice_review_rates.png")
    plot_slice_error_rates(slice_report, fig_dir_p / "slice_error_rates.png")

    dq = basic_quality_report(df, spec)
    (out_dir_p / "data_quality.json").write_text(json.dumps(dq, indent=2), encoding="utf-8")

    manifest = build_run_manifest(
        df,
        input_path=input_path,
        random_state=random_state,
        calibration_method=calibration_method,
        target_coverage=recommend_target_coverage,
    )
    (out_dir_p / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "metrics": metrics,
        "policy": policy,
        "baseline_metrics": baseline_metrics,
        "policy_variants": policy_variants,
        "slice_summary": slice_summary,
        "run_manifest": manifest,
        "outputs_dir": str(out_dir_p),
        "figures_dir": str(fig_dir_p),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Underwriting Decision Safety Lab pipeline")
    p.add_argument("--input", required=True, help="Path to loan approval CSV")
    p.add_argument("--out-dir", default="outputs", help="Output directory")
    p.add_argument("--figures-dir", default="reports/figures", help="Figures directory")
    p.add_argument(
        "--calibration",
        default="sigmoid",
        choices=["sigmoid", "isotonic"],
        help="Calibration method",
    )
    p.add_argument(
        "--target-coverage", type=float, default=0.70, help="Target auto-decision coverage"
    )
    args = p.parse_args()

    res = run(
        input_path=args.input,
        out_dir=args.out_dir,
        figures_dir=args.figures_dir,
        calibration_method=args.calibration,
        recommend_target_coverage=float(args.target_coverage),
    )

    print("\nDone! Underwriting report card created.")
    print(f"Outputs: {args.out_dir}")
    print(f"Figures: {args.figures_dir}")
    print("Primary model: LogisticRegression + calibration")
    print(
        f"Recommended threshold: {res['policy']['recommended_threshold']:.2f} (coverage≈{res['policy']['expected_coverage']:.2f})"  # noqa: E501
    )


if __name__ == "__main__":
    main()

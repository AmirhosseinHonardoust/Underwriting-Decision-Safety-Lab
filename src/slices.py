from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import expected_calibration_error

DEFAULT_CATEGORICAL_SLICE_COLUMNS = [
    "gender",
    "marital_status",
    "employment_status",
]

DEFAULT_NUMERIC_SLICE_COLUMNS = [
    "age",
    "credit_score",
    "annual_income",
]


def _safe_float(value: float | int | np.floating | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _safe_rate(mask: pd.Series | np.ndarray) -> float | None:
    arr = np.asarray(mask)
    if len(arr) == 0:
        return None
    return _safe_float(np.mean(arr))


def _bin_numeric_feature(series: pd.Series, bins: int = 4) -> pd.Series:
    """Create stable quantile-like bins for numeric slice reporting."""
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().sum() == 0:
        return pd.Series(["missing"] * len(series), index=series.index, dtype="object")

    if numeric.nunique(dropna=True) <= bins:
        out = numeric.astype("Int64").astype(str)
        return out.where(numeric.notna(), "missing")

    try:
        binned = pd.qcut(numeric, q=bins, duplicates="drop")
    except ValueError:
        binned = pd.cut(numeric, bins=bins, duplicates="drop")

    out = binned.astype(str)
    return out.where(numeric.notna(), "missing")


def add_numeric_slice_bins(
    df: pd.DataFrame,
    numeric_cols: Iterable[str] = DEFAULT_NUMERIC_SLICE_COLUMNS,
) -> pd.DataFrame:
    """Add human-readable numeric bins for slice reporting."""
    out = df.copy()

    for col in numeric_cols:
        if col in out.columns:
            out[f"{col}_band"] = _bin_numeric_feature(out[col])

    return out


def slice_metrics_for_group(
    group_df: pd.DataFrame,
    *,
    y_col: str = "y_true",
    p_col: str = "p_approve",
    pred_col: str = "pred_label",
    auto_col: str = "auto_decide",
    min_count: int = 20,
) -> dict[str, float | int | None]:
    """Compute decision-safety metrics for one data slice."""
    n = int(len(group_df))

    if n == 0:
        return {
            "n": 0,
            "observed_approval_rate": None,
            "mean_predicted_approval_probability": None,
            "auto_decision_rate": None,
            "review_rate": None,
            "auto_accuracy": None,
            "error_rate": None,
            "false_approval_rate": None,
            "false_rejection_rate": None,
            "ece": None,
            "is_small_slice": True,
        }

    y = group_df[y_col].astype(int)
    p = group_df[p_col].astype(float)
    pred = group_df[pred_col].astype(int)
    auto = group_df[auto_col].astype(bool)

    errors = pred != y
    false_approval = (pred == 1) & (y == 0)
    false_rejection = (pred == 0) & (y == 1)

    auto_accuracy = _safe_rate(pred[auto].to_numpy() == y[auto].to_numpy()) if auto.any() else None

    return {
        "n": n,
        "observed_approval_rate": _safe_float(y.mean()),
        "mean_predicted_approval_probability": _safe_float(p.mean()),
        "auto_decision_rate": _safe_rate(auto),
        "review_rate": _safe_rate(~auto),
        "auto_accuracy": auto_accuracy,
        "error_rate": _safe_rate(errors),
        "false_approval_rate": _safe_rate(false_approval),
        "false_rejection_rate": _safe_rate(false_rejection),
        "ece": (
            _safe_float(expected_calibration_error(y.to_numpy(), p.to_numpy(), n_bins=5))
            if n >= min_count and y.nunique(dropna=True) > 1
            else None
        ),
        "is_small_slice": bool(n < min_count),
    }


def build_slice_report(
    predictions: pd.DataFrame,
    *,
    slice_columns: Iterable[str] | None = None,
    min_count: int = 20,
) -> pd.DataFrame:
    """Build slice-level safety metrics from test predictions.

    Expected columns include y_true, p_approve, pred_label, and auto_decide.
    """
    required = {"y_true", "p_approve", "pred_label", "auto_decide"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing required columns: {sorted(missing)}")

    enriched = add_numeric_slice_bins(predictions)

    if slice_columns is None:
        slice_columns = [
            col for col in DEFAULT_CATEGORICAL_SLICE_COLUMNS if col in enriched.columns
        ] + [
            f"{col}_band"
            for col in DEFAULT_NUMERIC_SLICE_COLUMNS
            if f"{col}_band" in enriched.columns
        ]

    rows: list[dict] = []

    for col in slice_columns:
        if col not in enriched.columns:
            continue

        for value, group in enriched.groupby(col, dropna=False):
            metrics = slice_metrics_for_group(group, min_count=min_count)
            rows.append(
                {
                    "slice_feature": col,
                    "slice_value": "missing" if pd.isna(value) else str(value),
                    **metrics,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
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
            ]
        )

    report = pd.DataFrame(rows)
    return report.sort_values(["slice_feature", "slice_value"]).reset_index(drop=True)


def summarize_slice_report(report: pd.DataFrame) -> dict[str, float | int | None]:
    """Summarize largest slice disparities for quick report-card display."""
    if report.empty:
        return {
            "n_slices": 0,
            "max_auto_decision_rate_gap": None,
            "max_error_rate_gap": None,
            "max_ece_gap": None,
            "small_slice_count": 0,
        }

    def gap_for(metric: str) -> float | None:
        values = report[metric].dropna()
        if values.empty:
            return None
        return _safe_float(values.max() - values.min())

    return {
        "n_slices": int(len(report)),
        "max_auto_decision_rate_gap": gap_for("auto_decision_rate"),
        "max_error_rate_gap": gap_for("error_rate"),
        "max_ece_gap": gap_for("ece"),
        "small_slice_count": (
            int(report["is_small_slice"].sum()) if "is_small_slice" in report.columns else 0
        ),
    }


def save_slice_artifacts(report: pd.DataFrame, out_dir: str | Path) -> dict[str, Path]:
    """Save slice report artifacts in CSV and JSON formats."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    csv_path = out_path / "slice_report.csv"
    json_path = out_path / "slice_report.json"
    summary_path = out_path / "slice_summary.json"

    report.to_csv(csv_path, index=False)
    json_path.write_text(report.to_json(orient="records", indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summarize_slice_report(report), indent=2), encoding="utf-8")

    return {"csv": csv_path, "json": json_path, "summary": summary_path}

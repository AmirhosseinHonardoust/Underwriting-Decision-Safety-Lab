from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)


def _safe_float(value: float) -> float | None:
    value = float(value)
    return None if not np.isfinite(value) else value


def probability_metrics(y_true: np.ndarray, p_approve: np.ndarray) -> dict[str, float | None]:
    """Compute probability-quality metrics for binary approval probabilities."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p_approve, dtype=float)

    if len(y) != len(p):
        raise ValueError("y_true and p_approve must have the same length.")
    if len(y) == 0:
        raise ValueError("y_true and p_approve must not be empty.")
    if np.any((p < 0.0) | (p > 1.0)) or not np.isfinite(p).all():
        raise ValueError("p_approve must contain finite probabilities between 0 and 1.")

    out: dict[str, float | None] = {
        "brier": float(brier_score_loss(y, p)),
        "average_precision": float(average_precision_score(y, p)),
    }

    try:
        out["roc_auc"] = float(roc_auc_score(y, p))
    except ValueError:
        out["roc_auc"] = None

    return out


def classification_metrics_at_threshold(
    y_true: np.ndarray,
    p_approve: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute classification metrics after thresholding approval probabilities."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p_approve, dtype=float)
    pred = (p >= threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "threshold": float(threshold),
    }


def compute_baseline_metrics(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Evaluate simple baselines for context.

    These baselines help readers understand whether the trained model provides
    value beyond trivial decision policies.
    """
    baselines: dict[str, dict[str, Any]] = {}

    strategies = {
        "majority_class": "most_frequent",
        "empirical_prior": "prior",
        "stratified_random": "stratified",
    }

    for name, strategy in strategies.items():
        clf = DummyClassifier(strategy=strategy, random_state=42)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]

        metrics = {
            **probability_metrics(y_test, proba),
            **classification_metrics_at_threshold(y_test, proba, threshold=0.5),
            "strategy": strategy,
        }
        baselines[name] = metrics

    return baselines


def select_policy_variants(
    curve: pd.DataFrame,
    target_coverage: float = 0.70,
    min_quality_coverage: float = 0.25,
) -> dict[str, dict[str, float | str | None]]:
    """Create named abstention-policy options from a coverage curve.

    The curve is expected to contain threshold, coverage, accuracy, and f1.
    Variants are intended as decision-support options, not automatic policy.
    """
    required = {"threshold", "coverage", "accuracy", "f1"}
    missing = required - set(curve.columns)
    if missing:
        raise ValueError(f"Coverage curve missing required columns: {sorted(missing)}")

    c = curve.copy()
    c = c.replace([np.inf, -np.inf], np.nan)
    valid = c.dropna(subset=["coverage", "accuracy", "f1"])
    if valid.empty:
        raise ValueError("Coverage curve has no valid rows for policy selection.")

    def row_to_policy(row: pd.Series, description: str) -> dict[str, float | str | None]:
        return {
            "description": description,
            "threshold": float(row["threshold"]),
            "coverage": float(row["coverage"]),
            "accuracy_auto": _safe_float(row["accuracy"]),
            "f1_auto": _safe_float(row["f1"]),
        }

    target = valid.assign(distance=(valid["coverage"] - float(target_coverage)).abs()).sort_values(
        ["distance", "threshold"], ascending=[True, False]
    ).iloc[0]

    eligible_quality = valid[valid["coverage"] >= float(min_quality_coverage)]
    if eligible_quality.empty:
        eligible_quality = valid
    quality_first = eligible_quality.sort_values(["accuracy", "f1", "threshold"], ascending=[False, False, False]).iloc[0]

    high_coverage = valid.sort_values(["coverage", "accuracy"], ascending=[False, False]).iloc[0]
    balanced = valid.assign(balance_score=0.5 * valid["accuracy"] + 0.5 * valid["f1"]).sort_values(
        ["balance_score", "coverage"], ascending=[False, False]
    ).iloc[0]

    conservative_review = valid.sort_values(["threshold"], ascending=[False]).iloc[0]

    return {
        "target_coverage": row_to_policy(
            target,
            f"Closest policy to target auto-decision coverage {target_coverage:.2f}.",
        ),
        "quality_first": row_to_policy(
            quality_first,
            "Prioritizes high-quality auto-decisions while keeping some useful coverage.",
        ),
        "high_coverage": row_to_policy(
            high_coverage,
            "Maximizes automation coverage, accepting lower auto-decision quality if necessary.",
        ),
        "balanced": row_to_policy(
            balanced,
            "Balances auto-decision accuracy and F1.",
        ),
        "conservative_review": row_to_policy(
            conservative_review,
            "Routes more cases to review by requiring very high confidence for auto-decision.",
        ),
    }

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV


def _validate_binary_calibration_inputs(
    y_true: np.ndarray,
    p: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize inputs used by calibration helpers."""
    y = np.asarray(y_true, dtype=float).reshape(-1)
    prob = np.asarray(p, dtype=float).reshape(-1)

    if y.shape[0] != prob.shape[0]:
        raise ValueError("y_true and p must have the same length.")

    if y.shape[0] == 0:
        raise ValueError("y_true and p must not be empty.")

    if n_bins <= 0:
        raise ValueError("n_bins must be a positive integer.")

    if not np.isfinite(prob).all():
        raise ValueError("Predicted probabilities must be finite.")

    if np.any((prob < 0.0) | (prob > 1.0)):
        raise ValueError("Predicted probabilities must be between 0 and 1.")

    unique_labels = set(np.unique(y).tolist())
    if not unique_labels.issubset({0.0, 1.0}):
        raise ValueError("y_true must contain binary labels encoded as 0/1.")

    return y, prob


def calibration_bins(
    y_true: np.ndarray,
    p: np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    """Return per-bin binary probability calibration statistics.

    The predicted probability `p` is interpreted as the probability of the
    positive class, here approval. For each probability bin, the observed rate
    must therefore be `mean(y_true)`, not classification accuracy.
    """
    y, prob = _validate_binary_calibration_inputs(y_true, p, n_bins)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []

    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        mask = (prob >= lo) & (prob < hi) if i < n_bins - 1 else (prob >= lo) & (prob <= hi)

        count = int(np.sum(mask))
        if count == 0:
            rows.append(
                {
                    "bin_index": i,
                    "bin_lower": lo,
                    "bin_upper": hi,
                    "count": 0,
                    "weight": 0.0,
                    "mean_predicted_probability": np.nan,
                    "observed_positive_rate": np.nan,
                    "absolute_error": np.nan,
                }
            )
            continue

        mean_predicted = float(np.mean(prob[mask]))
        observed_rate = float(np.mean(y[mask]))
        abs_error = abs(observed_rate - mean_predicted)

        rows.append(
            {
                "bin_index": i,
                "bin_lower": lo,
                "bin_upper": hi,
                "count": count,
                "weight": float(count / len(prob)),
                "mean_predicted_probability": mean_predicted,
                "observed_positive_rate": observed_rate,
                "absolute_error": float(abs_error),
            }
        )

    return rows


def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Compute binary expected calibration error for positive-class probabilities.

    ECE is the weighted average absolute difference between each bin's mean
    predicted probability and the observed positive-class rate.
    """
    rows = calibration_bins(y_true, p, n_bins=n_bins)
    ece = 0.0

    for row in rows:
        if row["count"] == 0:
            continue
        ece += float(row["weight"]) * float(row["absolute_error"])

    return float(ece)


def calibrate(
    estimator: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    method: str = "sigmoid",
    cv: int = 3,
) -> CalibratedClassifierCV:
    cal = CalibratedClassifierCV(estimator, method=method, cv=cv)
    cal.fit(X_train, y_train)
    return cal

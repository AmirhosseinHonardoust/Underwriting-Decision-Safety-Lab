from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV


def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    p = np.asarray(p)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(p)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not np.any(mask):
            continue
        acc = np.mean(y_true[mask] == (p[mask] >= 0.5))
        conf = np.mean(p[mask])
        ece += (np.sum(mask) / n) * abs(acc - conf)

    return float(ece)


def calibrate(estimator, X_train, y_train, method: str = "sigmoid", cv: int = 3):
    cal = CalibratedClassifierCV(estimator, method=method, cv=cv)
    cal.fit(X_train, y_train)
    return cal

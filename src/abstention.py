from __future__ import annotations

from typing import Dict
    
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
  
  
def coverage_curve(y_true: np.ndarray, p_approve: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    p = np.asarray(p_approve)

    conf = np.maximum(p, 1.0 - p)
    pred = (p >= 0.5).astype(int)

    rows = []
    for t in thresholds:
        auto = conf >= t
        cov = float(np.mean(auto))
        if np.sum(auto) == 0:
            acc = float("nan")
            f1 = float("nan")
        else:
            acc = float(accuracy_score(y_true[auto], pred[auto]))
            f1 = float(f1_score(y_true[auto], pred[auto]))
        rows.append({"threshold": float(t), "coverage": cov, "accuracy": acc, "f1": f1})

    return pd.DataFrame(rows)


def recommend_threshold(curve: pd.DataFrame, target_coverage: float) -> Dict[str, float]:
    c = curve.copy()
    c["dist"] = (c["coverage"] - float(target_coverage)).abs()
    best = c.sort_values(["dist", "threshold"], ascending=[True, False]).iloc[0]
    return {
        "recommended_threshold": float(best["threshold"]),
        "expected_coverage": float(best["coverage"]),
        "expected_accuracy_auto": float(best["accuracy"]) if pd.notna(best["accuracy"]) else float("nan"),
        "expected_f1_auto": float(best["f1"]) if pd.notna(best["f1"]) else float("nan"),
    }

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
   

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, outpath: Path) -> None:
    _ensure_dir(outpath.parent)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Reject (0)", "Approve (1)"])
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title("Confusion Matrix (threshold=0.5)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_reliability_diagram(y_true: np.ndarray, p: np.ndarray, outpath: Path, n_bins: int = 10) -> None:
    _ensure_dir(outpath.parent)

    bins = np.linspace(0, 1, n_bins + 1)
    accs, confs = [], []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not np.any(mask):
            accs.append(np.nan)
            confs.append(np.nan)
            continue
        accs.append(np.mean(y_true[mask] == (p[mask] >= 0.5)))
        confs.append(np.mean(p[mask]))

    plt.figure(figsize=(7.2, 6.0))
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(confs, accs, marker="o")
    plt.xlabel("Mean predicted probability (approve)")
    plt.ylabel("Observed accuracy in bin")
    plt.title("Reliability Diagram (Calibration)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_probability_histograms(y_true: np.ndarray, p: np.ndarray, outpath: Path) -> None:
    _ensure_dir(outpath.parent)

    p = np.asarray(p)
    y_true = np.asarray(y_true)

    plt.figure(figsize=(7.8, 5.6))
    plt.hist(p[y_true == 1], bins=25, alpha=0.6, label="Approved (y=1)")
    plt.hist(p[y_true == 0], bins=25, alpha=0.6, label="Rejected (y=0)")
    plt.xlabel("Predicted probability of approval")
    plt.ylabel("Count")
    plt.title("Probability Histograms (separation + confidence)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_coverage_vs_performance(curve: pd.DataFrame, outpath: Path) -> None:
    _ensure_dir(outpath.parent)

    plt.figure(figsize=(7.8, 5.6))
    plt.plot(curve["coverage"], curve["accuracy"], marker="o", label="Accuracy (auto-decisions)")
    plt.plot(curve["coverage"], curve["f1"], marker="o", label="F1 (auto-decisions)")
    plt.xlabel("Coverage (fraction auto-decided)")
    plt.ylabel("Metric value")
    plt.title("Coverage vs Performance (Abstention tradeoff)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

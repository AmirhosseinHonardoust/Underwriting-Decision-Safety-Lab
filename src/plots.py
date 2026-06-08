from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, average_precision_score, confusion_matrix, precision_recall_curve

try:  # package import
    from .calibration import calibration_bins
except ImportError:  # script-style import fallback
    from calibration import calibration_bins


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
    plt.close(fig)


def plot_reliability_diagram(y_true: np.ndarray, p: np.ndarray, outpath: Path, n_bins: int = 10) -> None:
    """Plot predicted approval probability vs observed approval rate."""
    _ensure_dir(outpath.parent)

    rows = calibration_bins(y_true, p, n_bins=n_bins)
    plotted = [row for row in rows if row["count"] > 0]

    mean_predicted = [row["mean_predicted_probability"] for row in plotted]
    observed_rate = [row["observed_positive_rate"] for row in plotted]

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    ax.plot(mean_predicted, observed_rate, marker="o", label="Model")
    ax.set_xlabel("Mean predicted probability of approval")
    ax.set_ylabel("Observed approval rate in bin")
    ax.set_title("Reliability Diagram (Calibration)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_probability_histograms(y_true: np.ndarray, p: np.ndarray, outpath: Path) -> None:
    _ensure_dir(outpath.parent)

    p = np.asarray(p)
    y_true = np.asarray(y_true)

    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    ax.hist(p[y_true == 1], bins=25, alpha=0.6, label="Approved (y=1)")
    ax.hist(p[y_true == 0], bins=25, alpha=0.6, label="Rejected (y=0)")
    ax.set_xlabel("Predicted probability of approval")
    ax.set_ylabel("Count")
    ax.set_title("Probability Histograms (separation + confidence)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_precision_recall_curve(y_true: np.ndarray, p: np.ndarray, outpath: Path) -> None:
    _ensure_dir(outpath.parent)

    precision, recall, _ = precision_recall_curve(y_true, p)
    average_precision = average_precision_score(y_true, p)

    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    display = PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=average_precision)
    display.plot(ax=ax)
    ax.set_title("Precision-Recall Curve (Approval Class)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_coverage_vs_performance(curve: pd.DataFrame, outpath: Path) -> None:
    _ensure_dir(outpath.parent)

    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    ax.plot(curve["coverage"], curve["accuracy"], marker="o", label="Accuracy (auto-decisions)")
    ax.plot(curve["coverage"], curve["f1"], marker="o", label="F1 (auto-decisions)")
    ax.set_xlabel("Coverage (fraction auto-decided)")
    ax.set_ylabel("Metric value")
    ax.set_title("Coverage vs Performance (Abstention tradeoff)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close(fig)

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
         
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray


def make_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer:
    num_pipe = Pipeline([("scaler", StandardScaler())])
    cat_pipe = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, list(numeric_cols)),
            ("cat", cat_pipe, list(categorical_cols)),
        ],
        remainder="drop",
    )


def make_base_model(random_state: int = 42) -> LogisticRegression:
    return LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=random_state,
    )


def train_test_split_data(df: pd.DataFrame, target: str, *, test_size=0.25, random_state=42) -> SplitData:
    X = df.drop(columns=[target])
    y = df[target].astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return SplitData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def compute_binary_metrics(y_true: np.ndarray, proba_approve: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, proba_approve)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, proba_approve))
    except Exception:
        out["roc_auc"] = float("nan")
    return out

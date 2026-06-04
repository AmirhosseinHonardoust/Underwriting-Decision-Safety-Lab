from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


TARGET_CANDIDATES = ["loan_approved", "approved", "LoanApproved", "target", "label"]


@dataclass
class DataSpec:
    target: str
    id_cols: List[str]
    numeric_cols: List[str]
    categorical_cols: List[str]


def infer_spec(df: pd.DataFrame) -> DataSpec:
    cols = df.columns.tolist()
    target = None
    for c in TARGET_CANDIDATES:
        if c in cols:
            target = c
            break
    if target is None:
        raise ValueError(f"Could not find target column. Tried: {TARGET_CANDIDATES}")

    id_cols = [c for c in cols if c.lower() in {"applicant_id", "id"}]

    feature_cols = [c for c in cols if c != target and c not in id_cols]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    return DataSpec(target=target, id_cols=id_cols, numeric_cols=numeric_cols, categorical_cols=categorical_cols)


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def basic_quality_report(df: pd.DataFrame, spec: DataSpec) -> dict:
    out = {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "target": spec.target,
        "missing_rate_by_col": {c: float(df[c].isna().mean()) for c in df.columns},
        "n_unique_by_col": {c: int(df[c].nunique(dropna=True)) for c in df.columns},
    }

    hints = {}
    if "age" in df.columns:
        bad = (~df["age"].between(18, 100, inclusive="both")).sum()
        hints["age_out_of_range_count"] = int(bad)
    if "credit_score" in df.columns:
        bad = (~df["credit_score"].between(300, 850, inclusive="both")).sum()
        hints["credit_score_out_of_range_count"] = int(bad)
    out["plausibility_hints"] = hints
    return out

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import pandas as pd


class DataValidationError(ValueError):
    """Raised when underwriting input data is not valid for the pipeline."""


@dataclass(frozen=True)
class PlausibilityRule:
    """Allowed range (and integer-likeness) for a single underwriting column."""

    column: str
    minimum: float | None = None
    maximum: float | None = None
    integer_like: bool = False


DEFAULT_PLAUSIBILITY_RULES: tuple[PlausibilityRule, ...] = (
    PlausibilityRule("age", minimum=18, maximum=100),
    PlausibilityRule("credit_score", minimum=300, maximum=850),
    PlausibilityRule("annual_income", minimum=0),
    PlausibilityRule("loan_amount", minimum=0),
    PlausibilityRule("num_dependents", minimum=0, integer_like=True),
    PlausibilityRule("existing_loans_count", minimum=0, integer_like=True),
)


def _format_columns(columns: Sequence[str]) -> str:
    return ", ".join(repr(str(col)) for col in columns)


def validate_dataframe_structure(df: pd.DataFrame, *, min_rows: int = 2) -> None:
    """Validate universal dataframe structure before schema inference.

    This function checks conditions that can otherwise cause pandas ambiguity
    errors later, especially duplicate column names.
    """
    errors: list[str] = []

    if df.empty:
        errors.append("dataframe is empty")

    if len(df) < min_rows:
        errors.append(f"dataframe must contain at least {min_rows} rows")

    if len(df.columns) == 0:
        errors.append("dataframe must contain at least one column")

    if df.columns.duplicated().any():
        duplicates = sorted({str(col) for col in df.columns[df.columns.duplicated(keep=False)]})
        errors.append(f"duplicate column names found: {duplicates}")

    blank_columns = [str(col) for col in df.columns if str(col).strip() == ""]
    if blank_columns:
        errors.append("column names must not be blank")

    all_null_columns: list[str] = []
    unsupported_columns: list[str] = []

    for idx, col in enumerate(df.columns):
        series = df.iloc[:, idx]
        if series.isna().all():
            all_null_columns.append(str(col))
        if pd.api.types.is_complex_dtype(series.dtype):
            unsupported_columns.append(str(col))

    if all_null_columns:
        errors.append(f"columns cannot be entirely missing: {all_null_columns}")

    if unsupported_columns:
        errors.append(f"unsupported complex-valued columns: {unsupported_columns}")

    if errors:
        raise DataValidationError("Invalid input data: " + "; ".join(errors))


def validate_target_column(df: pd.DataFrame, target: str) -> None:
    """Validate that the target exists, is complete, and is binary 0/1."""
    if target not in df.columns:
        raise DataValidationError(f"target column {target!r} is missing")

    y = df[target]
    if y.isna().any():
        raise DataValidationError(f"target column {target!r} contains missing values")

    values = set(y.dropna().unique().tolist())
    allowed = {0, 1}
    if not values.issubset(allowed):
        raise DataValidationError(
            f"target column {target!r} must be binary 0/1; found values: {sorted(map(str, values))}"
        )

    if y.nunique(dropna=True) < 2:
        raise DataValidationError(f"target column {target!r} must contain both classes 0 and 1")


def validate_numeric_features(df: pd.DataFrame, numeric_cols: Iterable[str]) -> None:
    """Validate that inferred numeric columns are numeric and finite/non-missing."""
    bad_type: list[str] = []
    missing: list[str] = []

    for col in numeric_cols:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            bad_type.append(str(col))
            continue
        if df[col].isna().any():
            missing.append(str(col))

    if bad_type:
        raise DataValidationError(
            f"numeric columns must contain numeric values: {_format_columns(bad_type)}"
        )

    if missing:
        raise DataValidationError(
            f"numeric columns contain missing values: {_format_columns(missing)}"
        )


def validate_categorical_features(df: pd.DataFrame, categorical_cols: Iterable[str]) -> None:
    """Validate categorical columns for missing values."""
    missing = [str(col) for col in categorical_cols if col in df.columns and df[col].isna().any()]
    if missing:
        raise DataValidationError(
            f"categorical columns contain missing values: {_format_columns(missing)}"
        )


def validate_plausible_ranges(
    df: pd.DataFrame,
    rules: Iterable[PlausibilityRule] = DEFAULT_PLAUSIBILITY_RULES,
) -> None:
    """Validate common underwriting feature ranges when those columns exist."""
    errors: list[str] = []

    for rule in rules:
        if rule.column not in df.columns:
            continue

        series = df[rule.column]
        if not pd.api.types.is_numeric_dtype(series):
            errors.append(f"{rule.column!r} must be numeric")
            continue

        if rule.minimum is not None:
            count = int((series < rule.minimum).sum())
            if count:
                errors.append(f"{rule.column!r} has {count} values below {rule.minimum}")

        if rule.maximum is not None:
            count = int((series > rule.maximum).sum())
            if count:
                errors.append(f"{rule.column!r} has {count} values above {rule.maximum}")

        if rule.integer_like:
            non_integer = int(((series.dropna() % 1) != 0).sum())
            if non_integer:
                errors.append(f"{rule.column!r} has {non_integer} non-integer values")

    if errors:
        raise DataValidationError("Invalid underwriting feature ranges: " + "; ".join(errors))


def validate_underwriting_dataframe(
    df: pd.DataFrame,
    *,
    target: str,
    numeric_cols: Iterable[str],
    categorical_cols: Iterable[str],
    min_rows: int = 20,
) -> None:
    """Run all training-pipeline validation checks for underwriting data."""
    validate_dataframe_structure(df, min_rows=min_rows)
    validate_target_column(df, target)
    validate_numeric_features(df, numeric_cols)
    validate_categorical_features(df, categorical_cols)
    validate_plausible_ranges(df)

    feature_cols = [col for col in list(numeric_cols) + list(categorical_cols) if col in df.columns]
    if not feature_cols:
        raise DataValidationError("no usable feature columns detected")

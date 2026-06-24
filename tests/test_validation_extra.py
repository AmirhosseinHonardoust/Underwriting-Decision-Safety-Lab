from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from underwriting.validation import (
    DataValidationError,
    validate_categorical_features,
    validate_dataframe_structure,
    validate_numeric_features,
    validate_plausible_ranges,
    validate_target_column,
    validate_underwriting_dataframe,
)


def _valid_frame(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "age": rng.integers(21, 65, n),
            "annual_income": rng.integers(20_000, 150_000, n),
            "credit_score": rng.integers(300, 850, n),
            "num_dependents": rng.integers(0, 5, n),
            "existing_loans_count": rng.integers(0, 5, n),
            "gender": rng.choice(["Male", "Female"], n),
            "loan_approved": rng.integers(0, 2, n),
        }
    )


class StructureValidationTests(unittest.TestCase):
    def test_empty_frame_raises(self) -> None:
        with self.assertRaises(DataValidationError):
            validate_dataframe_structure(pd.DataFrame())

    def test_too_few_rows_raises(self) -> None:
        with self.assertRaises(DataValidationError):
            validate_dataframe_structure(pd.DataFrame({"a": [1]}), min_rows=2)

    def test_blank_column_name_raises(self) -> None:
        with self.assertRaises(DataValidationError):
            validate_dataframe_structure(pd.DataFrame({" ": [1, 2], "b": [1, 2]}))

    def test_all_null_column_raises(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": [None, None]})
        with self.assertRaises(DataValidationError):
            validate_dataframe_structure(df)


class TargetValidationTests(unittest.TestCase):
    def test_missing_target_raises(self) -> None:
        with self.assertRaises(DataValidationError):
            validate_target_column(pd.DataFrame({"a": [0, 1]}), "loan_approved")

    def test_target_with_missing_value_raises(self) -> None:
        with self.assertRaises(DataValidationError):
            validate_target_column(pd.DataFrame({"t": [0, 1, None]}), "t")

    def test_non_binary_target_raises(self) -> None:
        with self.assertRaises(DataValidationError):
            validate_target_column(pd.DataFrame({"t": [0, 1, 2]}), "t")

    def test_single_class_target_raises(self) -> None:
        with self.assertRaises(DataValidationError):
            validate_target_column(pd.DataFrame({"t": [1, 1, 1]}), "t")


class FeatureValidationTests(unittest.TestCase):
    def test_non_numeric_numeric_column_raises(self) -> None:
        df = pd.DataFrame({"x": ["a", "b", "c"]})
        with self.assertRaises(DataValidationError):
            validate_numeric_features(df, ["x"])

    def test_numeric_column_with_missing_raises(self) -> None:
        df = pd.DataFrame({"x": [1.0, None, 3.0]})
        with self.assertRaises(DataValidationError):
            validate_numeric_features(df, ["x"])

    def test_numeric_validation_skips_absent_columns(self) -> None:
        validate_numeric_features(pd.DataFrame({"x": [1, 2]}), ["not_here"])  # no raise

    def test_categorical_with_missing_raises(self) -> None:
        df = pd.DataFrame({"c": ["a", None, "b"]})
        with self.assertRaises(DataValidationError):
            validate_categorical_features(df, ["c"])


class PlausibilityValidationTests(unittest.TestCase):
    def test_value_below_minimum_raises(self) -> None:
        df = _valid_frame()
        df.loc[0, "age"] = 5  # below 18
        with self.assertRaises(DataValidationError):
            validate_plausible_ranges(df)

    def test_value_above_maximum_raises(self) -> None:
        df = _valid_frame()
        df.loc[0, "credit_score"] = 9999  # above 850
        with self.assertRaises(DataValidationError):
            validate_plausible_ranges(df)

    def test_non_integer_count_raises(self) -> None:
        df = _valid_frame().astype({"num_dependents": float})
        df.loc[0, "num_dependents"] = 1.5
        with self.assertRaises(DataValidationError):
            validate_plausible_ranges(df)

    def test_non_numeric_range_column_raises(self) -> None:
        df = _valid_frame().astype({"age": object})
        df.loc[0, "age"] = "old"
        with self.assertRaises(DataValidationError):
            validate_plausible_ranges(df)


class CombinedValidationTests(unittest.TestCase):
    def test_no_usable_feature_columns_raises(self) -> None:
        df = _valid_frame()
        with self.assertRaises(DataValidationError):
            validate_underwriting_dataframe(
                df,
                target="loan_approved",
                numeric_cols=["absent_a"],
                categorical_cols=["absent_b"],
                min_rows=5,
            )


if __name__ == "__main__":
    unittest.main()

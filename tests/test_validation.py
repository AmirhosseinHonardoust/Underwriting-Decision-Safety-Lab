from __future__ import annotations

import unittest

import pandas as pd

from src.data import infer_spec
from src.validation import (
    DataValidationError,
    validate_dataframe_structure,
    validate_target_column,
    validate_underwriting_dataframe,
)


class ValidationTests(unittest.TestCase):
    def _valid_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "applicant_id": range(1, 31),
                "age": [25, 35, 45, 55, 60, 28, 39, 41, 52, 33] * 3,
                "annual_income": [
                    45000,
                    60000,
                    75000,
                    90000,
                    120000,
                    52000,
                    68000,
                    71000,
                    85000,
                    58000,
                ]
                * 3,
                "loan_amount": [
                    10000,
                    15000,
                    20000,
                    25000,
                    30000,
                    12000,
                    17000,
                    21000,
                    26000,
                    14000,
                ]
                * 3,
                "credit_score": [620, 680, 720, 760, 810, 640, 700, 730, 770, 660] * 3,
                "num_dependents": [0, 1, 2, 0, 3, 1, 2, 0, 1, 2] * 3,
                "existing_loans_count": [0, 1, 1, 2, 0, 1, 2, 0, 1, 1] * 3,
                "gender": [
                    "Male",
                    "Female",
                    "Male",
                    "Female",
                    "Male",
                    "Female",
                    "Male",
                    "Female",
                    "Male",
                    "Female",
                ]
                * 3,
                "marital_status": [
                    "Single",
                    "Married",
                    "Married",
                    "Single",
                    "Divorced",
                    "Single",
                    "Married",
                    "Married",
                    "Single",
                    "Divorced",
                ]
                * 3,
                "employment_status": [
                    "Employed",
                    "Self-Employed",
                    "Employed",
                    "Employed",
                    "Self-Employed",
                    "Employed",
                    "Employed",
                    "Self-Employed",
                    "Employed",
                    "Employed",
                ]
                * 3,
                "loan_approved": [0, 1, 1, 1, 1, 0, 1, 1, 1, 0] * 3,
            }
        )

    def test_valid_underwriting_dataframe_passes(self) -> None:
        df = self._valid_df()
        spec = infer_spec(df)
        validate_underwriting_dataframe(
            df,
            target=spec.target,
            numeric_cols=spec.numeric_cols,
            categorical_cols=spec.categorical_cols,
        )

    def test_duplicate_columns_raise_before_schema_inference(self) -> None:
        df = pd.DataFrame([[1, 2], [3, 4]], columns=["age", "age"])
        with self.assertRaisesRegex(DataValidationError, "duplicate column"):
            validate_dataframe_structure(df)

    def test_missing_target_raises_clear_error(self) -> None:
        df = self._valid_df().drop(columns=["loan_approved"])
        with self.assertRaisesRegex(ValueError, "target column"):
            infer_spec(df)

    def test_non_binary_target_raises(self) -> None:
        df = self._valid_df()
        df.loc[0, "loan_approved"] = 2
        with self.assertRaisesRegex(DataValidationError, "binary"):
            validate_target_column(df, "loan_approved")

    def test_missing_numeric_value_raises(self) -> None:
        df = self._valid_df()
        df.loc[0, "credit_score"] = None
        spec = infer_spec(df)
        with self.assertRaisesRegex(DataValidationError, "numeric columns contain missing"):
            validate_underwriting_dataframe(
                df,
                target=spec.target,
                numeric_cols=spec.numeric_cols,
                categorical_cols=spec.categorical_cols,
            )

    def test_plausibility_range_raises(self) -> None:
        df = self._valid_df()
        df.loc[0, "age"] = 140
        spec = infer_spec(df)
        with self.assertRaisesRegex(DataValidationError, "age"):
            validate_underwriting_dataframe(
                df,
                target=spec.target,
                numeric_cols=spec.numeric_cols,
                categorical_cols=spec.categorical_cols,
            )


if __name__ == "__main__":
    unittest.main()

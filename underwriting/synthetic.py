"""Generate a synthetic loan-approval dataset matching the production schema.

This lets the full pipeline run end-to-end with no external data. The target has
a learnable relationship to credit score, income, loan amount, and employment, so
metrics are meaningful, and every column stays within the validated plausibility
ranges (see ``underwriting.validation``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GENDERS = ("Male", "Female")
MARITAL_STATUSES = ("Single", "Married", "Divorced")
EMPLOYMENT_STATUSES = ("Employed", "Self-employed", "Unemployed")


def generate_synthetic(n_rows: int = 1000, random_state: int = 0) -> pd.DataFrame:
    """Return a synthetic dataset with the same schema as the real input CSV."""
    if n_rows < 1:
        raise ValueError("n_rows must be a positive integer")

    rng = np.random.default_rng(random_state)

    age = rng.integers(21, 65, size=n_rows)
    gender = rng.choice(GENDERS, size=n_rows)
    marital_status = rng.choice(MARITAL_STATUSES, size=n_rows)
    annual_income = rng.integers(20_000, 150_000, size=n_rows)
    loan_amount = rng.integers(5_000, 50_000, size=n_rows)
    credit_score = rng.integers(300, 850, size=n_rows)
    num_dependents = rng.integers(0, 5, size=n_rows)
    existing_loans_count = rng.integers(0, 5, size=n_rows)
    employment_status = rng.choice(EMPLOYMENT_STATUSES, size=n_rows)

    # Latent approval signal: standardized contributions plus noise.
    z = (
        1.8 * (credit_score - 575) / 275
        + 1.0 * (annual_income - 85_000) / 65_000
        - 1.0 * (loan_amount - 27_500) / 22_500
        - 0.5 * existing_loans_count
        + np.where(employment_status == "Unemployed", -1.0, 0.3)
        + rng.normal(0.0, 0.5, size=n_rows)
    )
    prob = 1.0 / (1.0 + np.exp(-z))
    loan_approved = (rng.random(n_rows) < prob).astype(int)

    return pd.DataFrame(
        {
            "applicant_id": np.arange(1, n_rows + 1),
            "age": age,
            "gender": gender,
            "marital_status": marital_status,
            "annual_income": annual_income,
            "loan_amount": loan_amount,
            "credit_score": credit_score,
            "num_dependents": num_dependents,
            "existing_loans_count": existing_loans_count,
            "employment_status": employment_status,
            "loan_approved": loan_approved,
        }
    )


def main() -> None:
    """Command-line entry point for generating a synthetic dataset."""
    p = argparse.ArgumentParser(description="Generate a synthetic loan-approval dataset")
    p.add_argument("--out", required=True, help="Output CSV path")
    p.add_argument("--rows", type=int, default=1000, help="Number of rows to generate")
    p.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    args = p.parse_args()

    df = generate_synthetic(n_rows=args.rows, random_state=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} synthetic rows to {out}")


if __name__ == "__main__":
    main()

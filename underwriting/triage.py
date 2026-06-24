"""Pure decision-safe triage logic, extracted from the Streamlit app for testing.

The app is a thin UI shell; the decision rule and feature-row assembly live here so
they can be unit-tested independently of Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

APPLICANT_COLUMNS = (
    "age",
    "gender",
    "marital_status",
    "annual_income",
    "loan_amount",
    "credit_score",
    "num_dependents",
    "existing_loans_count",
    "employment_status",
)


@dataclass(frozen=True)
class TriageDecision:
    """Outcome of applying the abstention rule to a single approval probability."""

    p_approve: float
    confidence: float
    auto_decide: bool
    decision: str


def triage_decision(p_approve: float, threshold: float) -> TriageDecision:
    """Apply the confidence-threshold abstention rule to one approval probability.

    Confidence is ``max(p, 1 - p)``; the case is auto-decided when confidence is at
    or above the threshold, otherwise it is routed to manual review.
    """
    if not 0.0 <= p_approve <= 1.0:
        raise ValueError("p_approve must be in [0, 1]")
    confidence = max(p_approve, 1.0 - p_approve)
    auto = confidence >= threshold
    return TriageDecision(
        p_approve=float(p_approve),
        confidence=float(confidence),
        auto_decide=bool(auto),
        decision="AUTO-DECIDE" if auto else "REVIEW",
    )


def applicant_to_frame(
    *,
    age: int,
    gender: str,
    marital_status: str,
    annual_income: float,
    loan_amount: float,
    credit_score: int,
    num_dependents: int,
    existing_loans_count: int,
    employment_status: str,
) -> pd.DataFrame:
    """Build a single-row applicant DataFrame with the model's feature columns."""
    return pd.DataFrame(
        [
            {
                "age": age,
                "gender": gender,
                "marital_status": marital_status,
                "annual_income": annual_income,
                "loan_amount": loan_amount,
                "credit_score": credit_score,
                "num_dependents": num_dependents,
                "existing_loans_count": existing_loans_count,
                "employment_status": employment_status,
            }
        ]
    )

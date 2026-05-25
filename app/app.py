import sys
import json
import base64
from pathlib import Path
                    
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib
   
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import run as run_pipeline  # noqa: E402


def _img_tile(path: Path, caption: str, height_px: int = 340) -> None:
    """Fixed-height image tile for clean grids."""
    if not path.exists():
        st.warning(f"Missing figure: {path.name}")
        return
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    st.markdown(
        f"""
        <div style="border:1px solid rgba(49,51,63,0.15); border-radius:14px; padding:10px; background:white;">
          <img src="data:image/png;base64,{b64}" style="width:100%; height:{height_px}px; object-fit:contain; display:block;" />
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(caption)


st.set_page_config(page_title="Underwriting Decision Safety Lab", layout="wide")
st.title("Underwriting Decision Safety Lab")
st.caption("Calibration + abstention + decision-safe policy UI for loan approval.")

DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "loanapproval.csv"
OUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"

with st.sidebar:
    st.header("Pipeline")
    uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
    input_path = st.text_input("Or CSV path", value=str(DEFAULT_INPUT))
    target_cov = st.slider("Target auto-decision coverage", 0.10, 0.95, 0.70, 0.05)
    calibration = st.selectbox("Calibration method", ["sigmoid", "isotonic"], index=0)
    fig_height = st.slider("Figure tile height", 260, 560, 360, 10)
    run_btn = st.button("Run / Refresh", type="primary")

effective_input = Path(input_path)
if uploaded is not None:
    tmp = PROJECT_ROOT / "data" / "raw" / "uploaded.csv"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(uploaded.getbuffer())
    effective_input = tmp

if run_btn:
    with st.spinner("Training + calibrating + evaluating..."):
        run_pipeline(
            input_path=str(effective_input),
            out_dir=str(OUT_DIR),
            figures_dir=str(FIG_DIR),
            calibration_method=str(calibration),
            recommend_target_coverage=float(target_cov),
        )
    st.success("✅ Done! Outputs regenerated.")

metrics_path = OUT_DIR / "metrics_overall.json"
policy_path = OUT_DIR / "abstention_policy.json"
preds_path = OUT_DIR / "test_predictions.csv"
curve_path = OUT_DIR / "coverage_curve.csv"
dq_path = OUT_DIR / "data_quality.json"
model_path = OUT_DIR / "model.joblib"

if not metrics_path.exists():
    st.info("Run the pipeline from the sidebar to generate the report card.")
    st.stop()

metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
policy = json.loads(policy_path.read_text(encoding="utf-8")) if policy_path.exists() else {}
preds = pd.read_csv(preds_path) if preds_path.exists() else pd.DataFrame()
curve = pd.read_csv(curve_path) if curve_path.exists() else pd.DataFrame()
dq = json.loads(dq_path.read_text(encoding="utf-8")) if dq_path.exists() else {}

tab_report, tab_curve, tab_triage, tab_quality, tab_notes = st.tabs(
    ["Report Card", "Coverage Curve", "Triage UI", "Data Quality", "Notes"]
)

with tab_report:
    st.subheader("Report card (test set)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f'{metrics.get("accuracy", float("nan")):.3f}')
    c2.metric("F1", f'{metrics.get("f1", float("nan")):.3f}')
    c3.metric("ROC-AUC", f'{metrics.get("roc_auc", float("nan")):.3f}')
    c4.metric("ECE", f'{metrics.get("ece", float("nan")):.4f}')
    c5.metric("Brier", f'{metrics.get("brier", float("nan")):.4f}')

    if policy:
        st.markdown("### Recommended abstention policy")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Threshold (confidence)", f'{policy["recommended_threshold"]:.2f}')
        p2.metric("Expected coverage", f'{policy["expected_coverage"]:.2f}')
        p3.metric("Auto accuracy", f'{policy["expected_accuracy_auto"]:.3f}')
        p4.metric("Auto F1", f'{policy["expected_f1_auto"]:.3f}')

    st.subheader("Figures")
    r1 = st.columns(2, gap="large")
    with r1[0]:
        _img_tile(FIG_DIR / "confusion_matrix.png", "Confusion matrix (baseline threshold=0.5)", height_px=fig_height)
    with r1[1]:
        _img_tile(FIG_DIR / "reliability_diagram.png", "Reliability diagram (calibration)", height_px=fig_height)

    r2 = st.columns(2, gap="large")
    with r2[0]:
        _img_tile(FIG_DIR / "coverage_vs_performance.png", "Coverage vs performance (abstention tradeoff)", height_px=fig_height)
    with r2[1]:
        _img_tile(FIG_DIR / "probability_histograms.png", "Probability histograms (confidence separation)", height_px=fig_height)

    if policy:
        with st.expander("Policy JSON"):
            st.json(policy)

with tab_curve:
    st.subheader("Coverage frontier")
    st.caption("As threshold increases, coverage drops but auto-decision quality typically improves.")
    if not curve.empty:
        fig = px.line(curve, x="coverage", y=["accuracy", "f1"], markers=True)
        st.plotly_chart(fig, width="stretch")

        fig2 = px.line(curve, x="threshold", y=["coverage", "accuracy", "f1"], markers=True)
        st.plotly_chart(fig2, width="stretch")
    else:
        st.info("Coverage curve not found.")

with tab_triage:
    st.subheader("Interactive decision-safe triage")
    st.caption("Enter an applicant profile -> see probability, confidence, and whether the system should auto-decide or review.")

    if not model_path.exists():
        st.info("Model not found. Run pipeline to create outputs/model.joblib.")
        st.stop()

    payload = joblib.load(model_path)
    model = payload["model"]

    with st.form("applicant_form"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", min_value=18, max_value=100, value=35, step=1)
        gender = c2.selectbox("Gender", ["Male", "Female"])
        marital = c3.selectbox("Marital status", ["Single", "Married", "Divorced"])

        c4, c5, c6 = st.columns(3)
        income = c4.number_input("Annual income", min_value=0, value=80000, step=1000)
        loan_amount = c5.number_input("Loan amount", min_value=0, value=25000, step=500)
        credit_score = c6.number_input("Credit score", min_value=300, max_value=850, value=700, step=1)

        c7, c8, c9 = st.columns(3)
        dependents = c7.number_input("Num dependents", min_value=0, value=1, step=1)
        existing_loans = c8.number_input("Existing loans count", min_value=0, value=1, step=1)
        employment = c9.selectbox("Employment status", ["Employed", "Unemployed", "Self-Employed"])

        st.markdown("#### Decision threshold")
        thr = st.slider(
            "Confidence threshold (higher = safer, more review)",
            0.50,
            0.99,
            float(policy.get("recommended_threshold", 0.70)),
            0.01,
        )

        submitted = st.form_submit_button("Evaluate", type="primary")

    if submitted:
        row = pd.DataFrame(
            [{
                "age": age,
                "gender": gender,
                "marital_status": marital,
                "annual_income": income,
                "loan_amount": loan_amount,
                "credit_score": credit_score,
                "num_dependents": dependents,
                "existing_loans_count": existing_loans,
                "employment_status": employment,
            }]
        )
        p = float(model.predict_proba(row)[:, 1][0])
        conf = max(p, 1.0 - p)
        auto = conf >= float(thr)

        d1, d2, d3 = st.columns(3)
        d1.metric("p(approve)", f"{p:.3f}")
        d2.metric("confidence", f"{conf:.3f}")
        d3.metric("Decision", "AUTO-DECIDE" if auto else "REVIEW")

        st.plotly_chart(
            px.bar(
                x=["reject", "approve"],
                y=[1.0 - p, p],
                labels={"x": "class", "y": "probability"},
                title="Predicted probabilities",
            ),
            width="stretch",
        )

        st.caption(f"Rule: confidence = max(p, 1-p). Review when confidence < {thr:.2f}.")

with tab_quality:
    st.subheader("Data quality (quick checks)")
    if dq:
        st.json(dq)
        st.markdown(
            """
### Why this matters
Loan models are sensitive to:
- missingness patterns (non-random missingness is a signal)
- implausible values (age/credit score ranges)
- category drift (new employment types)

Use these checks as a lightweight "quality gate" before trusting results.
            """.strip()
        )
    else:
        st.info("No data quality report found (outputs/data_quality.json).")

with tab_notes:
    st.markdown(
        """
## Decision safety notes
- **Accuracy != trust.** A model can be accurate but overconfident.
- **ECE** is calibration error (lower is better).
- **Coverage** is a product metric: abstain too much and you lose usability; abstain too little and you increase risk.

## How to make it production-grade
- Add fairness slice audits: calibration and error rates by gender/age/employment.
- Add monitoring: score distribution drift, approval rate drift, coverage drift.
- Add a cost model: false-approvals vs false-rejections vs review cost.
- Retraining policy: trigger when calibration or coverage shifts materially.
        """.strip()
    )

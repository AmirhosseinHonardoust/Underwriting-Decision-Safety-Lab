<div align="center">

# Underwriting Decision Safety Lab

**Calibration + abstention + decision-safe policy UI for loan approval.**  
Turn model scores into **actions you can defend**: *auto-approve / auto-reject / send-to-review*.

<img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-blue" />
<img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-App-FF4B4B" />
<img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-ML-F7931E" />
<img alt="pandas" src="https://img.shields.io/badge/pandas-Data%20Frames-150458" />
<img alt="License" src="https://img.shields.io/badge/License-MIT-green" />
<img alt="Status" src="https://img.shields.io/badge/Status-Research%20Prototype-yellow" />

</div>

---

## Why this repo exists

Most ML projects stop at **“here’s the AUC”**. Underwriting can’t.

In lending, a prediction is only useful if it can be turned into a *decision policy* with:
- **Calibrated probabilities** (a “0.90 approve” should mean ~90% approval correctness under similar conditions)
- **Abstention** (a *review* path for uncertain cases)
- **Coverage tradeoffs** (how many cases you can safely auto-decide without breaking quality)
- **Decision-safe UI** (so the user sees confidence + what rule triggered the outcome)

This lab implements a full, end-to-end workflow:
1. Train a baseline loan approval model
2. Calibrate predicted probabilities
3. Build a **coverage frontier** (threshold ↔ auto-decision rate ↔ quality)
4. Recommend a **defensible threshold policy**
5. Provide an interactive Streamlit app for **triage** + **reporting**

> **Disclaimer:** This is a data science lab / portfolio project. It is not financial advice and not a production underwriting system.

---

## What you get

### Pipeline outputs
- `outputs/metrics_overall.json` | model metrics on the test split (accuracy, F1, ROC-AUC, ECE, Brier, …)
- `outputs/abstention_policy.json` | recommended threshold + expected coverage + expected “auto” quality
- `outputs/test_predictions.csv` | per-row probabilities + labels (used by the triage UI)
- `outputs/coverage_curve.csv` | threshold sweep results (coverage vs performance)

### Figures
Placed in `reports/figures/`:

- Confusion Matrix (baseline threshold)
- Coverage vs Performance (abstention tradeoff)
- Probability Histograms (separation + confidence)
- Reliability Diagram (calibration)

### Streamlit dashboard
Tabs:
- **Report Card**
- **Coverage Curve**
- **Triage UI**
- **Data Quality**
- **Notes**

---

## Dataset

This lab uses Kaggle’s **Loan Approval Dataset** (`loanapproval.csv`).  
Key columns (as shown in the Data Quality tab):

- `applicant_id` (unique ID)
- `age` (numeric)
- `gender` (categorical)
- `marital_status` (categorical)
- `annual_income` (numeric)
- `loan_amount` (numeric)
- `credit_score` (numeric)
- `num_dependents` (numeric)
- `existing_loans_count` (numeric)
- `employment_status` (categorical)
- `loan_approved` (target, 0/1)

### Quick sanity check (what the Data Quality screen shows)
- **Rows:** 1000  
- **Columns:** 11  
- **Missingness:** 0 in all columns (in this dataset snapshot)
- **Target balance:** approval is majority (typical of curated demo datasets)

> Even if missingness is 0 here, the Data Quality tab is important: underwriting models are extremely sensitive to *quiet schema drift* (new employment types, score ranges shifting, etc).

---

## Project structure

```

underwriting-decision-safety-lab/
├─ app/
│  └─ app.py                    # Streamlit dashboard
├─ data/
│  └─ raw/
│     └─ loanapproval.csv
├─ outputs/                     # generated JSON/CSV artifacts
├─ reports/
│  └─ figures/                  # generated PNG charts
└─ src/
   ├─ pipeline.py               # main pipeline entrypoint
   ├─ clean.py                  # cleaning + schema normalization
   ├─ train.py                  # model training
   ├─ calibrate.py              # sigmoid/isotonic calibration
   ├─ abstention.py             # threshold sweep + policy recommendation
   ├─ metrics.py                # ECE/Brier/etc
   └─ plots.py                  # figure generation

````

---

## How to run

### 1) Create environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
````

### 2) Run the pipeline (generates outputs + figures)

```bash
python -m src.pipeline --input data/raw/loanapproval.csv
```

You should see something like:

* Done!
* Outputs: `outputs/`
* Figures: `reports/figures/`

### 3) Launch the Streamlit app

```bash
streamlit run app/app.py
```

---

## Dashboard tour (screens + “how to read it”)

Below are the dashboard screens you shared. Each section explains:

* what the screen is answering
* how to interpret the results
* what actions you can take next

---

# 1) Report Card tab

**Goal:** Answer “Is the model *good enough* to consider auto-decisions and what’s the safe default policy?”

### What you see

* Top-level test metrics: **Accuracy, F1, ROC-AUC, ECE, Brier**
* A **recommended abstention policy** (threshold and expected coverage)
* A 2×2 figure grid:

  * Confusion matrix
  * Reliability diagram
  * Coverage vs performance
  * Probability histograms
  
<img width="1698" height="895" alt="Screenshot 2026-02-20 at 13-25-28 Underwriting Decision Safety Lab" src="https://github.com/user-attachments/assets/085df6b5-5f59-48e3-a2be-2b0d2647fddc" />

---

## Metrics explained (with underwriting meaning)

### Accuracy

“How often do we predict correctly overall?”
**Why it’s not enough:** A model can be accurate but dangerously overconfident.

### F1

Balances precision and recall (especially useful when class imbalance exists).
**Underwriting interpretation:** Helps you avoid “approve everything” or “reject everything” behavior.

### ROC-AUC

Ranking quality: “Do approved cases generally get higher scores than rejected cases?”
**Underwriting interpretation:** Strong AUC helps, but it doesn’t guarantee good threshold decisions.

### ECE (Expected Calibration Error)

Measures how far confidence deviates from reality across probability bins.

* If the model says ~0.8 approval 100 times, about 80 of them should actually be approved.
* High ECE means probabilities are not trustworthy without calibration.

### Brier score

Mean squared error of probabilistic predictions.

* Lower is better.
* Penalizes confident wrong predictions heavily.

---

# 2) Confusion matrix figure

<img width="1224" height="1044" alt="confusion_matrix" src="https://github.com/user-attachments/assets/16e2fbc5-2f85-4074-b64e-69049eaec957" />

### What it answers

“At a baseline threshold (often 0.5), what kinds of mistakes are we making?”

### How to read it

* Rows = true label
* Columns = predicted label
* Diagonal = correct predictions
* Off-diagonals = errors:

  * **False approvals** (predict approve but actually reject), risk/credit loss
  * **False rejections** (predict reject but actually approve), opportunity cost/customer friction

### Why it’s only the starting point

Underwriting typically *does not* operate at a single fixed threshold.
The point of this repo is to move from “0.5 classifier” → **policy**:

* conservative auto-approves
* conservative auto-rejects
* everything else → review

---

# 3) Reliability diagram (Calibration)

<img width="1296" height="1080" alt="reliability_diagram" src="https://github.com/user-attachments/assets/a428d1a2-d31e-42fa-83c3-98a285ca27bf" />

### What it answers

“Can we trust predicted probabilities as probabilities?”

### How to read it

* X-axis: predicted probability (binned)
* Y-axis: observed accuracy / empirical frequency
* The diagonal line = perfect calibration

  * points above line: underconfident (reality > confidence)
  * points below line: overconfident (confidence > reality)

### Why calibration is critical here

Abstention rules depend on confidence thresholds like:

* auto-decide only if confidence ≥ 0.85
  If probabilities are miscalibrated, “0.85” is not meaningful.

### What good looks like

* Points close to diagonal across mid-to-high probability regions
* Especially important near the decision threshold you’ll deploy

---

# 4) Probability histograms (separation + confidence)

<img width="1404" height="1007" alt="probability_histograms" src="https://github.com/user-attachments/assets/6403ca2a-c072-4d18-8277-c37f04d85304" />

### What it answers

“Does the model separate approvals from rejections and where does uncertainty live?”

### How to read it

* Two overlapping histograms:

  * Approved (y=1)
  * Rejected (y=0)
* If distributions are well-separated, the model can confidently auto-decide more cases.
* If they overlap heavily near the middle, you’ll need more abstention/review.

### Underwriting insights you can pull from this

* A large mass near **1.0** for approvals suggests strong “safe approve” region.
* A spread-out rejection distribution suggests rejections are harder to identify or less consistent.
* The overlap zone is your review queue candidate.

---

# 5) Coverage vs Performance (Abstention tradeoff)

<img width="1404" height="1007" alt="coverage_vs_performance" src="https://github.com/user-attachments/assets/fb255814-1433-44c5-92ad-a61258abd426" />

### What it answers

“How much quality do we gain if we abstain more?”

### Definitions

* **Coverage:** fraction of cases the system auto-decides
* **Auto-performance:** accuracy/F1 measured only on auto-decided cases
* As threshold increases:

  * coverage usually decreases
  * auto-quality usually increases

### How to use it (practical workflow)

1. Decide a target coverage (e.g., 70% auto-decide)
2. Choose the confidence threshold that achieves it
3. Verify auto-quality is acceptable
4. Review queue size becomes (1 - coverage)

### Common trap

High auto-accuracy is easy if you abstain on everything difficult.
So you must always report **both coverage + quality** together.

---

# 6) Coverage Curve tab

**Goal:** Make the coverage frontier interactive and easy to inspect.

<img width="1694" height="747" alt="Screenshot 2026-02-20 at 13-25-42 Underwriting Decision Safety Lab" src="https://github.com/user-attachments/assets/3415238e-0f1f-4a9c-9c4f-56992d6b95c8" />

### What you’re typically looking for

* A “knee” in the curve: a region where a small reduction in coverage buys a big jump in quality
* Stability: avoid thresholds where tiny changes cause big swings
* A defensible operating point:

  * “At threshold 0.85 we auto-decide ~70% with ~0.98 auto-accuracy”

---

# 7) Triage UI tab (Decision-safe demo)

**Goal:** Show *how an underwriter or analyst would experience the model*.

<img width="1707" height="838" alt="Screenshot 2026-02-20 at 13-25-58 Underwriting Decision Safety Lab" src="https://github.com/user-attachments/assets/e5bbdd15-3783-4e47-98e0-83f8e4afb17a" />

### What it does

* You enter applicant features (age, income, loan amount, credit score, etc.)
* The app outputs:

  * `p(approve)`
  * a confidence measure (often max probability or margin)
  * a decision: **AUTO-DECIDE** or **REVIEW**
  * a bar chart of class probabilities

### The decision-safe rule (core concept)

Instead of saying “approve” because p=0.71, the UI says:

* **AUTO-DECIDE** when confidence ≥ threshold
* **REVIEW** otherwise

This makes the system defensible:

* you can explain what confidence threshold you chose
* you can estimate workload (review volume)
* you can monitor drift (coverage changing over time)

### Why this is better than raw predictions

A raw probability without a policy invites misuse:

* different teams interpret it differently
* thresholds get chosen ad hoc
* you lose traceability for “why was this decision made?”

---

# 8) Data Quality tab (Quick checks)

**Goal:** Catch problems before you trust metrics.

<img width="475" height="891" alt="Screenshot 2026-02-20 at 13-26-15 Underwriting Decision Safety Lab" src="https://github.com/user-attachments/assets/0b503770-0fc8-4e24-9eab-e1220722a43b" />

### What this tab should include (and why)

Even in clean demo datasets, underwriting systems in the wild break due to:

#### Missingness drift

* income missing for a new channel
* employment status missing for a partner integration

#### Plausibility violations

* credit scores outside expected range
* negative loan amounts
* impossible ages

#### Category drift

* new `employment_status` values
* changes in marital status encoding

### Why this matters for decision safety

A policy like “auto-decide above 0.85” assumes your feature distribution is similar to training.
Data quality checks are the “trust gate” before policy application.

---

# 9) Notes tab (Interpretation + production guidance)

<img width="575" height="402" alt="Screenshot 2026-02-20 at 13-26-21 Underwriting Decision Safety Lab" src="https://github.com/user-attachments/assets/8b062c56-6408-405d-b8ed-c9b9b29fbd05" />

### The important message

* **Accuracy ≠ trust**
* **ECE is calibration error**
* **Coverage is a product metric** (review queue size is not free)

### What “production-grade” means here

A real system should add:

* fairness slice audits (calibration + error rates by gender/age/employment)
* monitoring: score drift, approval-rate drift, coverage drift
* cost-aware policy: false approvals vs false rejections vs review cost
* retraining triggers when calibration degrades

---

## Recommended abstention policy (what it means)

The app shows a recommended policy (example from your report card screen):

* **Threshold (confidence):** 0.85
* **Expected coverage:** 0.70
* **Auto accuracy:** 0.977
* **Auto F1:** 0.986

Interpretation:

* The system auto-decides ~70% of applicants.
* The remaining ~30% go to human review.
* Auto-decided cases are high-confidence, so quality is high.
* This is **not “cheating”**, it is a conscious design decision that turns ML into a safe workflow.

---

## How to extend this lab

### 1) Two-sided policy (approve + reject + review)

Right now, many prototypes use one confidence threshold. Underwriting often benefits from:

* auto-approve if p(approve) ≥ T_approve
* auto-reject if p(approve) ≤ T_reject
* else review

This reduces review load while controlling risk.

### 2) Cost-aware optimization

Replace “maximize accuracy” with:

* cost(false approval) >> cost(false rejection)
* cost(review) as a workload term

Then choose thresholds that minimize expected cost.

### 3) Fairness-aware reporting

Add slice dashboards:

* ECE by gender
* error rate by age band
* approval rate by employment status
* coverage by subgroup

### 4) Monitoring playbook

Track weekly:

* score distribution drift
* coverage drift
* approval-rate drift
* calibration drift (ECE moving)

---

## Troubleshooting

### “My Streamlit warnings mention `use_container_width`”

Newer Streamlit versions prefer:

* `width="stretch"` instead of `use_container_width=True`

If you see deprecation warnings, update your `st.plotly_chart(...)` and `st.image(...)` calls accordingly.

### “Figures look too small / layout weird”

Ensure:

* `st.set_page_config(layout="wide")`
* Use consistent containers/columns
* Use `width="stretch"` for charts/images in Streamlit

---

## Credits

* Dataset: https://www.kaggle.com/datasets/amineipad/loan-approval-dataset
* Tools: pandas, scikit-learn, Streamlit, matplotlib/plotly

# Insurance Claim Settlement Bias Analysis

A Streamlit dashboard for Settlement Officers to detect, diagnose, and quantify bias in insurance claim decisions.

---

## Features

| Section | What it does |
|---|---|
| **Overview** | Dataset snapshot, class distribution |
| **Descriptive Analysis** | Cross-tabulations vs policy status with Chi² tests |
| **Diagnostic Analysis** | Team/zone, age, income, and interaction-effect bias detection |
| **ML Models** | Feature engineering + KNN, Decision Tree, Random Forest, Gradient Boosted |
| **Model Evaluation** | Accuracy, Precision, Recall, F1, ROC curves, Confusion matrices, FP/FN % |
| **Findings** | Actionable bias findings and management recommendations |

---

## Quick Start (Local)

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/insurance-bias-analysis.git
cd insurance-bias-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Then open your browser at `http://localhost:8501` and upload `Insurance.csv`.

---

## Deploy on Streamlit Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Click **New app** → Select your repo → Set **Main file path** to `app.py`.
4. Click **Deploy**.
5. Once live, upload `Insurance.csv` via the sidebar.

---

## Dataset Columns

| Column | Description |
|---|---|
| `POLICY_NO` | Unique policy identifier |
| `PI_GENDER` | Policyholder gender (M/F) |
| `PI_AGE` | Policyholder age |
| `PI_ANNUAL_INCOME` | Annual income |
| `SUM_ASSURED` | Sum insured |
| `ZONE` | Handling team / zone |
| `PAYMENT_MODE` | Annual / Half-Yly / Quarterly / Monthly / Single |
| `EARLY_NON` | EARLY (claim within 2 yrs) or NON EARLY |
| `MEDICAL_NONMED` | Medical examination done or not |
| `PI_OCCUPATION` | Occupation category |
| `PI_STATE` | Indian state |
| `REASON_FOR_CLAIM` | Cause of death |
| `POLICY_STATUS` | **Target** – Approved Death Claim / Repudiate Death |

---

## Key Findings Summary

- **Team/Zone Bias** — Approval rates range 23–97% across teams (Chi² p < 0.0001)
- **Payment Mode Bias** — Single-pay: 89.9% vs Quarterly: 45.0% (44.9pp gap)
- **Medical Bias** — MEDICAL: 81.1% vs NON-MEDICAL: 66.4% (14.7pp gap)
- **Best Model** — Gradient Boosted (Test Acc 75.2%, AUC 0.790, lowest FP%)

---

## Tech Stack

- **Python 3.10+**
- **Streamlit** – dashboard framework
- **scikit-learn** – ML models
- **pandas / numpy** – data processing
- **matplotlib / seaborn** – visualisations
- **scipy** – statistical tests

---

## File Structure

```
insurance-bias-analysis/
├── app.py               ← Main Streamlit app
├── requirements.txt     ← Python dependencies
├── README.md            ← This file
└── Insurance.csv        ← Upload this in the app sidebar
```

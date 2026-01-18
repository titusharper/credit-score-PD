# credit-score-PD
The goal is to build a Probability of Default (PD) model that estimates the customer's risk of default (TARGET=1) as a probability.

# Credit Risk PD (Home Credit) — Modeling Pipeline

## Purpose
This repository contains the data science / modeling pipeline for predicting Probability of Default (PD) on the Home Credit dataset.  
Focus: feature engineering + LightGBM training + evaluation + submission generation.

> Note: The data analysis / EDA part of the project was done by a teammate. This repo focuses on the end-to-end modeling implementation.

---

## What’s inside
- **Feature engineering**
  - V1: base application + bureau + previous application aggregations
  - V2: adds raw tables (installments / POS / credit card / bureau balance) aggregations
  - V3 (best): memory-aware raw reads + recent-window features + engineered ratios/flags
- **Modeling**
  - LightGBM with Stratified K-Fold CV
  - OOF predictions + test predictions
- **Evaluation & artifacts**
  - AUC, PR-AUC, LogLoss, Brier, ECE
  - KS, Gini
  - Calibration table + reliability plot
  - Risk capture (Top-% capture) + plot
  - Group reports (gender / age bins)
- **Outputs**
  - `submission_best_v3.csv`
  - `metrics_best_v3.json`
  - plots + markdown report under `outputs/`

---

## Repo structure
  - src/credit_risk_pd/ reusable modules (data, features, modeling, evaluation)
  - scripts/ runnable scripts (main pipeline)
  - outputs/ generated artifacts (ignored by git)
  - data/ datasets (NOT tracked in git)

--- 

## Data setup (important)
Datasets are not included in this repository due to size.

- Raw CSVs:
  - `POS_CASH_balance.csv`
  - `installments_payments.csv`
  - `credit_card_balance.csv`
  - `bureau_balance.csv`
- Processed / cleaned CSVs:
  - `application_train_cleaned.csv`
  - `application_test_cleaned.csv`
  - `bureau_cleaned.csv`
  - `previous_application_cleaned.csv`

---

## Installation
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
---

## Results (V3 — best pipeline)

**Validation setup:** 5-fold Stratified CV (OOF = out-of-fold predictions)

**Primary metrics (OOF):**
- **ROC-AUC:** **0.7865**
  - Fold AUCs: 0.7840, 0.7950, 0.7830, 0.7892, 0.7817  
  - Fold mean ± std: **0.7866 ± 0.0049**
- **PR-AUC:** **0.2805**
- **LogLoss:** **0.2374**
- **Brier score:** **0.06594**
- **ECE (15 bins):** **0.00414**

**Calibration (post-hoc on OOF):**
- **Isotonic calibration** performed best vs Platt scaling:
  - **Brier:** **0.06583**
  - **LogLoss:** **0.23686**
  - **ECE (15 bins):** ~0.0

**Score statistics:**
- Target rate (train): **0.08073**
- Mean OOF prediction: **0.07795**
- Mean test prediction: **0.06896** (min: 0.00320, max: 0.72688)

**Generated artifacts (under `CRPD_OUTPUTS/`):**
- `submission_best_v3.csv`
- `metrics_best_v3.json`
- calibration table + reliability plot
- risk capture table + plot
- group reports (gender / age bins)
- summary markdown report

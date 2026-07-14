# Crisis-Risk-Model
Predictive modelling project focused on identifying community mental health patients at higher risk of crisis referral. Built a full pipeline covering data quality, feature engineering and model development, with outputs designed to support proactive caseload management.

## What this project does

1. **Audits data quality** — identifies missingness, duplicates, outliers, inconsistent coding, and a deliberate target leakage variable
2. **Engineers features** — derives clinically meaningful predictors (DNA rate, severity bands, engagement flags) from raw fields
3. **Selects features** — runs a multi-stage consensus pipeline (missingness filter → univariate tests → Random Forest importance)
4. **Trains and compares models** — Logistic Regression, Random Forest; evaluated via 5-fold stratified cross-validation
5. **Stratifies risk** — assigns patients to Low / Moderate / High / Very High risk tiers with observed crisis rates per tier
6. **Produces outputs** — confusion matrices, feature importance charts, risk tier summaries

---

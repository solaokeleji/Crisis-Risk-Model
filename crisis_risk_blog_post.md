# Predicting Crisis Referral Risk in Community Mental Health: A Machine Learning Walkthrough

One of the recurring challenges in community mental health is knowing who needs support before things reach a crisis point. Caseloads are large, resources are stretched, and the signals that a patient is deteriorating aren't always obvious until it's too late. This project explores whether a predictive model, built on data that teams are already collecting, can help identify patients at higher risk of crisis referral within 90 days.

The dataset used here is synthetic and pseudonymised, covering 2,040 records across 28 fields: demographics, clinical scores, service use patterns, and a binary outcome indicating whether each patient had a crisis referral within 90 days of their index point. This post walks through the full pipeline, from data quality issues to risk stratification and shares what the model found.

---

## Step 1: Catching Problems Before They Corrupt the Model

The first thing that came up before any analysis, was a field called `Post_Index_Crisis_Flag`. It turned out to be a near-perfect predictor of the outcome. That's because it _encodes post-index information_: it tells you what happened after the point you're trying to predict from. Including it would have produced a model that looked brilliantly accurate in testing and completely failed in deployment. It was removed immediately.

Beyond that, the dataset had a fairly typical set of real-world data quality problems:

- **40 duplicate Patient IDs** — exact duplicates, dropped safely by keeping the first occurrence
- **Inconsistent Gender coding** — values like "M", "male", "Male" and "F", "female", "Female" all coexisting, standardised to Male/Female/Unknown
- **Invalid values** — ages of -4, IMD Decile values of 0 or 11 (outside the valid 1–10 range), BMI values below 10 and above the 99th percentile
- **170 negative values** in `Days_Since_Last_Contact` — a date recording error, since a contact date cannot be after the index date; set to missing
- **Missingness across multiple fields**, ranging from 5% (Gender) to 21% (Smoking Status), with Discharge Date appearing 70% missing — though that's expected, since a blank means the case is still open

Missing categorical values in Ethnicity, Referral Source, and Primary Diagnosis were grouped into an 'Unknown' category rather than imputed, since assuming a value for these fields could introduce bias. Numeric fields were imputed using the median, keeping the process robust to the outliers that were already present.

---

## Step 2: Understanding the Data Before Building Anything

With a clean dataset, the next step was understanding what was actually in it before touching a model.

**The outcome was imbalanced.** About 75% of patients had no crisis referral within 90 days; 25% did. That 3:1 ratio isn't extreme, but it's enough to cause problems if you evaluate models on accuracy alone, a model that predicts "no crisis" for everyone would be 75% accurate and completely useless.

**No single feature strongly predicts the outcome.** Correlation analysis confirmed this: no individual variable had a strong linear relationship with crisis referral. This is expected: mental health crisis is a multi-factor outcome, shaped by the interaction between clinical severity, service engagement, social circumstances, and history. That's precisely why a model is more useful than a simple rule.

**Some patterns were visible in the group-level means.** Patients who went on to crisis referral had notably higher average Previous Admissions (0.86 vs 0.45), more A&E visits in the prior 12 months (1.52 vs 0.89), and longer average gaps since their last contact (214 vs 189 days). None of these alone are predictive, but together they start to sketch a picture.

---

## Step 3: Engineering Useful Features

Raw fields don't always translate cleanly into model inputs. Several new features were derived from the existing data:

**Clinically meaningful categories** replaced raw numeric scores. PHQ-9 and GAD-7 scores were bucketed into validated severity bands (Minimal / Mild / Moderate / Severe) rather than fed in as continuous values. Age was grouped into bands. IMD Decile was grouped into Low, Medium, and High deprivation categories. These transformations reduce noise, align with how clinicians actually interpret these scores, and avoid the model treating a difference between scores of 4 and 5 the same as a difference between 14 and 15.

**Engagement flags** were derived from service use patterns. A `Crisis_Contact_Flag` captured whether any crisis contact had occurred in the prior 12 months, a binary signal with strong face validity. A `DNA_Count_Flag` flagged patients whose Did Not Attend count was above the 40th percentile, as a proxy for disengagement.

**Open case status** was extracted from the Discharge Date field, a missing date means the case is still active, which carries different risk implications than a closed one.

Columns that were superseded by these engineered features were removed to avoid redundancy and multicollinearity.

---

## Step 4: Choosing the Right Evaluation Metric

Because the classes are imbalanced, **accuracy was not used as the primary metric**. A model that predicts "no crisis" for every patient would score 75% accuracy and be worthless. Instead, models were scored on **Average Precision (PR-AUC)**, which focuses on how well the model identifies true crisis cases without being swamped by the majority class.

Both models were trained inside a pipeline with stratified 5-fold cross-validated grid search, ensuring hyperparameter tuning didn't overfit to the training set.

---

## Step 5: Model Results

Two models were compared: **Random Forest** and **Logistic Regression**.

Random Forest overfit. Training accuracy was 1.00, a ceiling that immediately signals something is wrong. On the test set, despite using balanced class weights, it achieved a recall of just 0.019. That means it missed 98% of actual crisis cases. The model learned the majority class and effectively stopped trying.

Logistic Regression performed better on every metric that mattered:

| Metric                          | Logistic Regression | Random Forest |
| ------------------------------- | ------------------- | ------------- |
| PR-AUC                          | 0.441               | 0.377         |
| ROC-AUC                         | 0.694               | 0.638         |
| Test Recall (default threshold) | 0.612               | 0.019         |

The strongest predictors from the Logistic Regression coefficients were GAD-7 and PHQ-9 severity categories, age group, deprivation category, crisis contact history, DNA flag, and open case status, all features already collected in routine practice, which matters for adoption.

---

## Step 6: Choosing a Decision Threshold

The model produces a probability score for each patient. You then choose a threshold above which you flag someone as high risk. The default in most frameworks is 0.5, but that's rarely appropriate in clinical settings.

| Threshold | Sensitivity | Specificity | Precision |
| --------- | ----------- | ----------- | --------- |
| 0.25      | 100.0%      | 0.7%        | 25.9%     |
| 0.30      | 100.0%      | 6.4%        | 27.0%     |
| **0.35**  | **95.1%**   | **20.9%**   | **29.4%** |
| 0.40      | 86.4%       | 35.0%       | 31.6%     |
| 0.50      | 61.2%       | 68.0%       | 39.9%     |

A threshold of **0.35** was chosen. It catches 95.1% of actual crisis cases, meaning 19 in 20 patients who will go on to crisis referral are flagged. The trade-off is lower specificity: a meaningful number of patients are flagged who won't crisis-refer. In this context, that's the right trade-off. A missed crisis carries far greater cost than an unnecessary clinical review.

The threshold isn't fixed, it should be revisited with clinical stakeholders based on actual team capacity.

---

## Step 7: Risk Stratification

Applying the model across the full cohort, predicted probabilities were binned into four tiers:

| Tier      | Patients | % of Cohort |
| --------- | -------- | ----------- |
| Low       | 0        | 0%          |
| Medium    | 618      | 30.9%       |
| High      | 1,022    | 51.1%       |
| Very High | 360      | 18.0%       |

The absence of a Low risk group reflects the nature of the cohort, this is an active community mental health caseload, and baseline risk is elevated across the board. That finding itself is worth flagging to commissioners.

The **Very High tier of 360 patients** is where proactive outreach would have the greatest impact: same-week clinical review, care coordinator contact, or checking whether an open crisis plan is in place.

---

## What This Model Is and Isn't

It's a screening tool. It surfaces patterns in population-level data that would be difficult to spot manually across a large caseload. Used well, it supports clinical teams in prioritising attention, not replacing their judgement, but directing it.

It isn't a diagnostic tool, a substitute for clinical assessment, or a system that should ever trigger automated actions. Every output from this model should go to a clinician for review. A high risk score means "look at this person this week," not "this person will crisis-refer."

There are also things the model can't see. It knows about contacts and scores and demographics. It doesn't know what was said in an appointment last Tuesday, whether someone just moved into temporary accommodation, or whether a key relationship has broken down. Clinical context will always carry information the data doesn't.

---

## Recommendations

**For clinical teams:** Focus the Very High tier on proactive outreach. Review whether anyone in this group has an up-to-date crisis plan, is allocated a care coordinator, and has had contact within an appropriate timeframe.

**For data quality:** Referral Source is unknown for 25.6% of records, the most common value in the field. Better capture of this at source would sharpen future models without any change to the algorithm.

**For operational planning:** Days Since Last Contact and DNA rate are strong, actionable signals. These should be visible on caseload dashboards independently of any predictive model, because they matter whether or not you're using one.

**For governance:** Any move toward operational deployment would need a Data Protection Impact Assessment, clinical governance sign-off, and a fairness audit to check the model doesn't perform differently across demographic groups. That work needs to happen before anything goes live.

---

The full code for this project: data cleaning, feature engineering, model training, and risk stratification; is available on GitHub.

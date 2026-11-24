# Term Deposit Propensity Model: Pre-Call Targeting Without Data Leakage

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kunalpatel-08/Banking_Term_deposit_classification/blob/main/Propensity_bank_term_deposit_NO_DURATION.ipynb)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A deployable propensity model that predicts which bank customers are likely to subscribe to a term deposit **before** the call is made, so the bank can prioritise outreach and cut wasted calls.

---

## Table of Contents
- [Business Problem](#business-problem)
- [The Key Decision: Dropping `duration`](#the-key-decision-dropping-duration)
- [Dataset](#dataset)
- [Approach](#approach)
- [Results](#results)
- [Business Impact](#business-impact)
- [Threshold Selection](#threshold-selection)
- [Model Stability (PSI)](#model-stability-psi)
- [Limitations and Honest Caveats](#limitations-and-honest-caveats)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Author](#author)

---

## Business Problem

A bank runs telemarketing campaigns to sell term deposits. Only about **11.7%** of calls convert, so most agent time is spent on customers who will say no.

**Goal:** score every customer before dialling, so agents call the most promising customers first.

## The Key Decision: Dropping `duration`

The UCI Bank Marketing dataset includes `duration`, the length of the last call in seconds. It is the strongest predictor in the data, but it is only known **after the call ends**, so it cannot be used to decide whom to call.

| | With `duration` | Without `duration` (this model) |
|---|---|---|
| Test AUC | 0.9166 | **0.7776** |
| Gini | 0.8332 | **0.5553** |
| KS | 0.7038 | **0.4303** |
| Top-3 decile capture | 90.7% | **67.5%** |
| Deployable before the call? | No | **Yes** |

The notebook builds both versions. The with-duration model looks far better on paper, but it is not usable, since it answers "who already subscribed?" rather than "who will?". The 0.7776 AUC is the honest, deployable number.

## Dataset

- **Source:** [UCI Bank Marketing (bank-full)](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Size:** 45,211 customers, 16 input features (15 after dropping `duration`)
- **Target:** `y` (subscribed to a term deposit: yes/no)
- **Class balance:** 11.7% positive, about 7.5:1 imbalance

> The raw CSV is semicolon-delimited (`sep=';'`). The data file is not included in this repository, so download `bank-full.csv` from the UCI link above.

## Approach

1. **Data ingestion**: Google Colab with Google Drive, with an optional BigQuery path included in the notebook.
2. **EDA and leakage proof**: distribution and box plots showing `duration` separates the classes almost perfectly, which is the reason it is excluded.
3. **WoE / IV analysis**: Weight of Evidence and Information Value computed for all 15 pre-call features. Features with IV ≥ 0.02 are flagged as informative.
4. **Feature engineering**: 32 new features, giving **67 model inputs** after encoding. Highlights:
   - Recency score and engagement index from prior-contact history (`pdays`, `previous`)
   - Financial stress index (default + loan + housing)
   - Wealth readiness (log balance × education ÷ financial stress)
   - Cyclical month encoding (sin/cos), start/end-of-month flags
   - Age, balance and campaign-count transformations and interactions
5. **Split and imbalance handling**: stratified 70/15/15 train/validation/test split. **SMOTE is applied to the training set only**, and the scaler is fitted on training data only. Validation and test keep the natural 11.7% event rate.
6. **Model comparison**: Logistic Regression, Random Forest, XGBoost, LightGBM. XGBoost had the highest validation AUC and was selected.
7. **Banking-style evaluation**: AUC, Gini, KS, decile analysis, Rank-over-Break, cost-aware threshold tuning, and PSI.

## Results

### Test-set model comparison (natural imbalanced distribution, threshold 0.5)

| Model | AUC | Gini | KS | F1 | Precision | Recall |
|---|---|---|---|---|---|---|
| **XGBoost** | **0.7776** | **0.5553** | **0.4303** | 0.3575 | 0.5394 | 0.2673 |
| Random Forest | 0.7745 | 0.5490 | 0.4285 | 0.4090 | 0.3930 | 0.4262 |
| LightGBM | 0.7741 | 0.5482 | 0.4279 | 0.3493 | 0.5615 | 0.2535 |
| Logistic Regression | 0.6940 | 0.3880 | 0.2844 | 0.3159 | 0.3735 | 0.2736 |

The three tree-based models are close on AUC. Logistic Regression trails clearly, which suggests the signal is non-linear.

### Top features by Information Value

| Feature | IV | Strength |
|---|---|---|
| `poutcome` | 0.5146 | Very high (flagged for review, see caveats) |
| `month` | 0.4361 | Strong |
| `contact` | 0.3004 | Strong |
| `housing` | 0.1887 | Medium |
| `job` | 0.1557 | Medium |

## Business Impact

Decile analysis on the held-out test set (6,782 customers, 793 subscribers):

| Decile | Conversion rate | Rank-over-Break (lift) | Cumulative subscribers captured |
|---|---|---|---|
| 1 | 46.4% | 3.97x | 39.7% |
| 2 | 19.6% | 1.68x | 56.5% |
| 3 | 12.8% | 1.10x | 67.5% |

- Calling only the **top 30%** of scored customers reaches **67.5%** of all subscribers.
- The top decile converts at about **4x the base rate**.
- The maximum KS separation occurs at decile 3.

## Threshold Selection

The default 0.5 cutoff is poor for an imbalanced problem (recall of only 27%). The notebook compares thresholds, including one based on an illustrative cost assumption (a missed subscriber costs 10 units, a wasted call costs 1):

| Threshold | Precision | Recall | F1 | Total cost |
|---|---|---|---|---|
| 0.50 (default) | 0.539 | 0.267 | 0.358 | 5,991 |
| 0.30 | 0.423 | 0.450 | 0.436 | 4,847 |
| **0.27 (max F1)** | 0.405 | 0.483 | 0.441 | 4,662 |
| **0.11 (min cost)** | 0.238 | 0.714 | 0.357 | 4,086 |

- **0.27** is the better choice when the cost ratio is unknown.
- **0.11** is better when missing a subscriber is much more expensive than an extra call.
- The 10:1 cost ratio is an **assumption**. Replace it with the bank's real economics before using it.

## Model Stability (PSI)

| Comparison | PSI | Reading |
|---|---|---|
| Original train vs test scores | **0.0014** | Stable |
| SMOTE-resampled train vs test scores | 4.82 | Artifact of synthetic samples, not real drift |

PSI should be computed against the original (non-resampled) training distribution. The inflated value on the SMOTE data is a known artifact and is shown only for contrast.

## Limitations and Honest Caveats

- **Cross-validation is optimistic.** The 5-fold CV (AUC 0.97) was run on the SMOTE-resampled training set, where synthetic neighbours of a point can land in a different fold. Treat the **held-out test AUC (0.78)** as the real performance figure.
- **Threshold tuned on the test set.** Thresholds were swept on test predictions. For a production setting, select the threshold on the validation set and report final metrics on an untouched test set.
- **Illustrative cost ratio.** The 10:1 FN:FP cost is a placeholder, not bank data.
- **`poutcome` has high IV.** An IV above 0.5 is conventionally a prompt to check for leakage. Here it reflects genuine prior-campaign history (known before the call), but most of its signal comes from the "unknown" and "success" categories, so it is worth monitoring.
- **Single public dataset.** The data covers one Portuguese bank over a limited period, so results may not transfer to other markets without revalidation.
- **Call-savings figure is a simple estimate.** "Reach 67.5% of subscribers by calling 30% of customers" is a lift calculation. It does not account for call costs, deposit values, or campaign constraints.

## Repository Structure

```
Banking_Term_deposit_classification/
├── Propensity_bank_term_deposit_NO_DURATION.ipynb   # Full analysis and modelling notebook
└── README.md
```

The notebook is organised into 11 sections: setup, EDA, WoE/IV, feature engineering, split and SMOTE, training, evaluation, decile analysis, threshold tuning, PSI, and a final scorecard.

## How to Run

**Option 1: Google Colab (recommended)**
1. Click the **Open in Colab** badge at the top.
2. Download `bank-full.csv` from the [UCI repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) and upload it to your Drive or Colab session.
3. Update `file_path` in the data-loading cell, then run all cells.

**Option 2: Local**
```bash
git clone https://github.com/Kunalpatel-08/Banking_Term_deposit_classification.git
cd Banking_Term_deposit_classification
pip install numpy pandas scipy matplotlib seaborn scikit-learn imbalanced-learn xgboost lightgbm shap jupyter
jupyter notebook Propensity_bank_term_deposit_NO_DURATION.ipynb
```
Remove the `google.colab` drive-mount cell and set `file_path = 'bank-full.csv'`.

## Tech Stack

Python · pandas · NumPy · scikit-learn · imbalanced-learn (SMOTE) · XGBoost · LightGBM · matplotlib · seaborn · Google Colab · BigQuery (optional)

## Author

**Kunal Patel**
Mechanical engineer turned data analyst/scientist, with a background in manufacturing and a full-stack data science bootcamp, now focused on analytics and data science roles.

- GitHub: [@Kunalpatel-08](https://github.com/Kunalpatel-08)

---

*Dataset: Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. Decision Support Systems.*

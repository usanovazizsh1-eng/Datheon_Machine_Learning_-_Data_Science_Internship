# Telecom Customer Churn Prediction

An end-to-end machine learning pipeline built to predict customer churn using telecom subscription records. The project handles data cleaning, categorical feature encoding, class imbalance via SMOTE and cost-sensitive reweighting, cross-validated ensemble modeling (Random Forest and XGBoost), and ROC-AUC evaluation.

---

## Project Overview

Customer retention is critical for subscription-based telecommunication providers. Predicting churn allows companies to proactively reach out to at-risk subscribers with targeted retention offers.

This project implements:
1. **Data Preprocessing & Cleaning**: Converting non-numeric values, handling missing data, and stripping user identifiers.
2. **Feature Engineering & Encoding**: Dynamic `ColumnTransformer` applying `OneHotEncoder(drop='first')` to categorical variables while passing numeric features.
3. **Class Imbalance Mitigation**: Balancing training samples using **SMOTE** (Synthetic Minority Over-sampling Technique) for Random Forest and **`scale_pos_weight`** for XGBoost.
4. **Cross-Validated Ensembles**: Training `RandomForestClassifier` and `XGBClassifier` using 5-Fold Stratified Cross-Validation.
5. **Model Evaluation & Interpretation**: Evaluating performance via ROC-AUC curves ($> 80\%$ benchmark) and extracting top 10 feature importances.

---

## Dataset Specifications

* **Source**: Telco Customer Churn Dataset (IBM Community)
* **Total Records**: 7,043 customer accounts
* **Features**: 20 predictor variables (demographics, services subscribed, account details)
* **Target Variable**: `Churn` (`1` = Churned, `0` = Retained)
* **Class Ratio**: ~73.5% Retained ($N=5,174$) vs. ~26.5% Churned ($N=1,869$)

---

## Quick Start & Installation

### 1. Clone Repository & Install Dependencies

```bash
git clone [https://github.com/your-username/telco-customer-churn.git](https://github.com/your-username/telco-customer-churn.git)
cd telco-customer-churn
pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn
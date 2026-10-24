# California Housing Price Prediction & Regression Analysis

A end-to-end Machine Learning pipeline that predicts median house values in California using **Ordinary Least Squares (OLS)**, **Ridge ($L_2$)**, and **Lasso ($L_1$)** regression models. The project includes feature engineering, missing value imputation, categorical encoding, hyperparameter tuning via `GridSearchCV`, and diagnostic visualizations.

---

## Features & Project Architecture

- **Feature Engineering**: Constructs key interaction and ratio features (`rooms_per_household`, `bedrooms_per_room`, `population_per_household`, `income_location_interaction`) to capture spatial dynamics and scale-independent metrics.
- **Preprocessing Pipeline**: 
  - `SimpleImputer` (median strategy) for missing values in numerical columns.
  - `StandardScaler` for continuous feature scaling.
  - `OneHotEncoder` for categorical attributes (`ocean_proximity`).
  - Scikit-learn `ColumnTransformer` to eliminate data leakage.
- **Hyperparameter Optimization**: Uses 5-fold cross-validation with `GridSearchCV` to optimize regularization strength ($\alpha$) across Ridge and Lasso models.
- **Model Diagnostics**: Generates **Predicted vs. Actual** scatter plots and **Residual Distribution** curves.

---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/california-housing-regression.git
   cd california-housing-regression
   ```

2. **Install dependencies**:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

---

## Execution

Run the complete pipeline using Python:

```bash
python main.py
```

## Visual Diagnostic Key

1. **Predicted vs Actual Plots**:
   - The red dashed diagonal represents perfect prediction ($y = \hat{y}$).
   - The horizontal band at $500,000$ shows the price ceiling cap in the original dataset.
2. **Residual Distribution Curves**:
   - A symmetric, normal distribution centered at $0$ indicates an unbiased model with homoscedastic errors.
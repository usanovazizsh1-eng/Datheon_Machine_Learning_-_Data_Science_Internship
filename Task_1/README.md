# Iris Flower Classification Analysis

An end-to-end Machine Learning pipeline that classifies Iris flowers into three species (**Setosa**, **Versicolor**, **Virginica**) using **Logistic Regression** and **Decision Tree** models. The project covers Exploratory Data Analysis (EDA), feature scaling, model evaluation, and 2D decision boundary visualizations.

---

## Features & Project Architecture

* **Exploratory Data Analysis (EDA)**: Visualizes pairplots and distributions across sepal and petal features using Seaborn.
* **Preprocessing Pipeline**: Uses `StandardScaler` to scale continuous features for linear decision boundaries.
* **Model Training & Evaluation**: Compares Logistic Regression and Decision Tree classifiers using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix heatmaps.
* **Decision Boundary Visualization**: Uses `DecisionBoundaryDisplay` to map class boundaries in 2D space across petal measurements.

---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/iris-flower-classification.git
   cd iris-flower-classification
   ```

2. **Install dependencies**:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

---

## Execution

Run the complete classification and visualization pipeline using Python:

```bash
python main.py
```

## Visual Diagnostic Key

1. **Decision Boundaries**:
   * Shows spatial separation across species (`setosa`, `versicolor`, `virginica`) using petal dimensions.
2. **Confusion Matrices**:
   * Highlights true vs. predicted classifications to evaluate misclassifications between overlapping classes (`versicolor` vs `virginica`).
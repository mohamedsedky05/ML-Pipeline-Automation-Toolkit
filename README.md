# ML Pipeline Framework (MyOwnModule)

A reusable machine learning toolkit that automates preprocessing, outlier handling, model training, hyperparameter tuning, and evaluation for both **classification** and **regression** tasks.

This project was built as a personal framework to speed up experimentation while keeping workflows **clean, reproducible, and production-oriented**.

---

## ✨ Key Features

### ✅ Adaptive Outlier Handling (Custom Transformer)
- Implements a custom `OutlierTrimmer` transformer
- Automatically applies robust outlier trimming before training
- Supports common trimming strategies (e.g., IQR / percentile-based trimming)
- Designed to work smoothly inside `scikit-learn` pipelines

### ✅ Automated Preprocessing Pipelines
- Separate preprocessing for:
  - **Numerical features** (outliers, missing values, scaling)
  - **Categorical features** (missing values, encoding with unknown handling)
- Built using `sklearn.pipeline` and `ColumnTransformer`

### ✅ Model Training + Hyperparameter Optimization
- Ready-to-use evaluation workflow supporting:
  - **RandomizedSearchCV**
  - Cross-validation (classification & regression)
  - Multiple model comparison in one run

### ✅ Rich Evaluation Reports
- **Classification metrics**: Accuracy, Balanced Accuracy, Precision, Recall, F1, ROC-AUC
- **Regression metrics**: R², MAE, MSE, RMSE
- Produces clean tables that make it easy to compare models

---

## 📁 Project Contents

- `MyOwnModule.py`  
  Core utilities:
  - Custom transformers (e.g., `OutlierTrimmer`)
  - Preprocessing pipeline builders
  - Training + evaluation helpers
  - Model selection and reporting

- `Complete Case (Classification).html`  
  Full worked example (classification pipeline)

- `Complete Case (Regression).html`  
  Full worked example (regression pipeline)

---

## 🚀 Quick Start

### 1) Install dependencies

```bash
pip install -r requirements.txt
2) Run the examples
Open and review:

Complete Case (Classification).html

Complete Case (Regression).html

Or use the module in your own project:

import MyOwnModule as mom

# Example usage depends on your dataset and the helper functions provided
# (check the examples for a full workflow)
🧠 Why This Project?
Most ML projects repeat the same steps:

cleaning

preprocessing

handling outliers

trying multiple models

tuning hyperparameters

producing evaluation reports

This toolkit packages these steps into a reusable framework to improve:

speed of experimentation

code readability

consistency across projects

🛠 Tech Stack
Python

Pandas / NumPy

Scikit-learn


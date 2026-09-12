# Analytics Pipeline Module

## Overview
Comprehensive exploratory data analysis and predictive modeling on the Titanic dataset, including classification, imbalance handling, hyperparameter tuning, and regression.

## Data Source
- Titanic dataset loaded via `seaborn.load_dataset('titanic')` **exactly once** in `01_eda.ipynb`
- Saved deterministically to `titanic.csv` for offline use by all subsequent scripts
- **No additional network calls** are made after the initial load

## Notebooks & Scripts

### `01_eda.ipynb` — Exploratory Data Analysis
1. **Data Profiling**: `df.info()`, `df.describe()`, `df.shape`, exact missing value percentages
2. **Threshold-Based Cleaning**:
   - < 5% missing → Drop rows (e.g., `embarked` ~0.22%)
   - 5–30% missing → Impute with median (e.g., `age` ~19.87%)
   - > 30% missing → Encode 'Missing' as category (e.g., `deck` ~77.22%)
3. **Univariate Analysis**: IQR outlier detection for `age` and `fare`; skewness analysis (fare is right-skewed: Mean > Median > Mode)
4. **Bivariate Analysis**: Survival rates by sex, pclass, and sex×pclass; 6×6 correlation heatmap (columns: `survived`, `pclass`, `age`, `sibsp`, `parch`, `fare` — excludes `adult_male` and `alone`)
5. **Multivariate Story**: 4 charts with 2–4 sentence interpretations each
6. **Z-Score Standardization**: Exploratory check confirming mean ≈ 0 and std ≈ 1 post-scaling

### `02_modeling.ipynb` — Predictive Modeling
1. **Stratified Split**: 80/20 train/test split before any preprocessing (justified by ~38% minority class)
2. **Pipeline**: `ColumnTransformer` + `Pipeline` (SimpleImputer + StandardScaler/OneHotEncoder)
3. **Classifiers**: Logistic Regression, Decision Tree, Random Forest — all using identical train/test split
4. **Decision Tree Visual**: `plot_tree` with labeled features and classes → `charts/decision_tree.png`
5. **Metric Suite**: Confusion Matrix, Accuracy, Precision, Recall, F1, ROC-AUC for all 3 models
6. **Imbalance Comparison**: Baseline vs `class_weight='balanced'` vs SMOTE (via `imblearn.Pipeline` for zero leakage)
7. **GridSearchCV**: Tuning over `n_estimators`, `max_depth`, `max_features`; fresh `RandomForestClassifier(oob_score=True)` with best params
8. **Regression**: Multivariate linear regression predicting `fare` with MAE, RMSE, R², Adjusted R²; residual plot with heteroscedasticity conclusion
9. **Model Recommendation**: Justified by actual numerical metrics
10. **Serialization**: `best_pipeline.joblib` saved and verified

### `test_pipeline.py` — Pipeline Verification
Reloads `best_pipeline.joblib` and validates predictions on raw, unpreprocessed data (including NaN values to test imputation).

## Output Files
- `titanic.csv` — Deterministic offline dataset
- `best_pipeline.joblib` — Fitted classification pipeline
- `charts/correlation_heatmap.png` — 6×6 heatmap
- `charts/decision_tree.png` — Decision tree visualization
- `charts/boxplots_age_fare.png` — Outlier detection
- `charts/chart1–4_*.png` — Multivariate story charts
- `charts/residuals_plot.png` — Regression residual analysis

## Installation & Usage
```bash
cd analytics
pip install -r requirements.txt

# Run EDA (creates titanic.csv + charts)
python 01_eda.py  # or execute 01_eda.ipynb

# Run Modeling (creates best_pipeline.joblib + metrics)
python 02_modeling.py  # or execute 02_modeling.ipynb

# Verify pipeline
python test_pipeline.py
```

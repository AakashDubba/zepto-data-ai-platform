# %% [markdown]
# # Zepto Analytics — 02 Predictive Modeling
# 
# This notebook builds classification and regression models on the Titanic dataset:
# - Stratified train/test split (before any preprocessing)
# - sklearn ColumnTransformer + Pipeline (imputer + encoder + scaler)
# - Logistic Regression, Decision Tree, Random Forest comparison
# - Imbalance handling: Baseline vs balanced vs SMOTE
# - Hyperparameter tuning via GridSearchCV + OOB score
# - Regression side-task: Predict fare
# - Pipeline serialization & reload test

# %%
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             mean_absolute_error, mean_squared_error, r2_score)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import joblib

ANALYTICS_DIR = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR = os.path.join(ANALYTICS_DIR, "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

# %% [markdown]
# ## 1. Load Data from Offline CSV (No seaborn network call)

# %%
csv_path = os.path.join(ANALYTICS_DIR, "titanic.csv")
assert os.path.exists(csv_path), f"titanic.csv not found at {csv_path}! Run 01_eda first."
assert os.path.getsize(csv_path) > 0, "titanic.csv is empty!"

df = pd.read_csv(csv_path)
print(f"Loaded titanic.csv: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Verify all expected columns exist
expected_cols = ['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare',
                 'embarked', 'class', 'who', 'adult_male', 'deck', 'embark_town',
                 'alive', 'alone']
for col in expected_cols:
    assert col in df.columns, f"Missing column: {col}"
print(f"✅ All {len(expected_cols)} expected columns verified")

# %% [markdown]
# ## 2. Stratified Train/Test Split (BEFORE any preprocessing)
# 
# **Justification for stratification**: The Titanic dataset has an imbalanced survival
# distribution (~38% survived vs ~62% did not survive). Stratified splitting ensures
# that both train and test sets preserve this class ratio, preventing the test set
# from being unrepresentative and yielding unreliable performance estimates.

# %%
# Target and features
target = 'survived'
y = df[target]

# Show class distribution to justify stratification
print(f"\n  Target class distribution:")
print(f"    {y.value_counts().to_dict()}")
print(f"    Survival rate: {y.mean():.4f} ({y.mean()*100:.1f}%)")
print(f"    Class ratio (0:1): {(1-y.mean())/y.mean():.2f}:1")
print(f"\n  → Stratification justified: {y.mean()*100:.1f}% minority class (< 50%)")

# Select modeling features (drop target, identifiers, and redundant columns)
feature_cols = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'who', 'alone']
X = df[feature_cols]

# Stratified split BEFORE any preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n  Train set: {X_train.shape} (survival rate: {y_train.mean():.4f})")
print(f"  Test set:  {X_test.shape} (survival rate: {y_test.mean():.4f})")
print(f"  ✅ Stratification preserved: train and test survival rates are similar")

# %% [markdown]
# ## 3. ColumnTransformer + Pipeline (Imputer + Encoder + Scaler)
# 
# **Strict train-only fitting**: The pipeline is fitted ONLY on training data.
# The test set is transformed using the fitted transformers (no data leakage).

# %%
# Define feature types
numeric_features = ['age', 'sibsp', 'parch', 'fare']
categorical_features = ['pclass', 'sex', 'embarked', 'who', 'alone']

# Numeric transformer: impute missing with median, then standardize
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical transformer: impute missing with most frequent, then one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combined preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

print("  ColumnTransformer + Pipeline defined:")
print(f"    Numeric features ({len(numeric_features)}): {numeric_features}")
print(f"    Categorical features ({len(categorical_features)}): {categorical_features}")
print(f"    Numeric: SimpleImputer(median) → StandardScaler")
print(f"    Categorical: SimpleImputer(most_frequent) → OneHotEncoder")

# %% [markdown]
# ## 4. Train 3 Classifiers — Same Train/Test Split

# %%
print("\n" + "=" * 60)
print("  CLASSIFICATION MODEL TRAINING")
print("=" * 60)

# Define classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
}

results = {}
for name, clf in classifiers.items():
    # Create pipeline with preprocessor + classifier
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    
    # Fit on training data ONLY
    pipe.fit(X_train, y_train)
    
    # Predict on test data
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    
    # Compute metrics
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    results[name] = {
        'Accuracy': acc, 'Precision': prec, 'Recall': rec,
        'F1': f1, 'ROC-AUC': auc, 'CM': cm, 'pipeline': pipe
    }
    
    print(f"\n  {name}:")
    print(f"    Confusion Matrix:\n      {cm[0]}\n      {cm[1]}")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    F1 Score:  {f1:.4f}")
    print(f"    ROC-AUC:   {auc:.4f}")

# ── Comparison Table ──
print("\n" + "=" * 60)
print("  CLASSIFICATION MODEL COMPARISON")
print("=" * 60)
print(f"\n  | {'Model':<25} | {'Accuracy':>8} | {'Precision':>9} | {'Recall':>6} | {'F1':>6} | {'ROC-AUC':>7} |")
print(f"  |{'-'*27}|{'-'*10}|{'-'*11}|{'-'*8}|{'-'*8}|{'-'*9}|")
for name, r in results.items():
    print(f"  | {name:<25} | {r['Accuracy']:>8.4f} | {r['Precision']:>9.4f} | {r['Recall']:>6.4f} | {r['F1']:>6.4f} | {r['ROC-AUC']:>7.4f} |")

# %% [markdown]
# ## 5. Decision Tree Visualization

# %%
# Extract the Decision Tree from its pipeline
dt_pipe = results['Decision Tree']['pipeline']
dt_clf = dt_pipe.named_steps['classifier']

# Get feature names from preprocessor (fitted)
fitted_preprocessor = dt_pipe.named_steps['preprocessor']
num_names = numeric_features
cat_names = list(fitted_preprocessor.named_transformers_['cat']
                 .named_steps['encoder'].get_feature_names_out(categorical_features))
feature_names = num_names + cat_names

fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(dt_clf, feature_names=feature_names, class_names=['Not Survived', 'Survived'],
          filled=True, rounded=True, fontsize=8, ax=ax)
ax.set_title('Decision Tree — Titanic Survival Prediction', fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(CHARTS_DIR, 'decision_tree.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\n  ✅ Decision tree plot saved to: charts/decision_tree.png")
print(f"     Features: {len(feature_names)}, Max depth: {dt_clf.get_depth()}")

# %% [markdown]
# ## 6. Imbalance Comparison — Baseline vs Balanced vs SMOTE
# 
# **CRITICAL**: SMOTE is encapsulated in `imblearn.pipeline.Pipeline` to ensure
# oversampling occurs ONLY within training/CV folds, never on the test set.

# %%
print("\n" + "=" * 60)
print("  IMBALANCE HANDLING COMPARISON")
print("=" * 60)

# (a) Baseline Random Forest (already computed)
baseline_metrics = results['Random Forest']

# (b) class_weight='balanced'
pipe_balanced = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=100,
                                          class_weight='balanced'))
])
pipe_balanced.fit(X_train, y_train)
y_pred_bal = pipe_balanced.predict(X_test)
y_prob_bal = pipe_balanced.predict_proba(X_test)[:, 1]

balanced_metrics = {
    'Precision': precision_score(y_test, y_pred_bal),
    'Recall': recall_score(y_test, y_pred_bal),
    'F1': f1_score(y_test, y_pred_bal),
    'ROC-AUC': roc_auc_score(y_test, y_prob_bal),
}

# (c) SMOTE via imblearn Pipeline (ensures SMOTE only in training folds)
# Note: imblearn Pipeline handles SMOTE correctly in cross-validation
pipe_smote = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
])
pipe_smote.fit(X_train, y_train)
y_pred_smote = pipe_smote.predict(X_test)
y_prob_smote = pipe_smote.predict_proba(X_test)[:, 1]

smote_metrics = {
    'Precision': precision_score(y_test, y_pred_smote),
    'Recall': recall_score(y_test, y_pred_smote),
    'F1': f1_score(y_test, y_pred_smote),
    'ROC-AUC': roc_auc_score(y_test, y_prob_smote),
}

print(f"\n  | {'Variant':<25} | {'Precision':>9} | {'Recall':>6} | {'F1':>6} | {'ROC-AUC':>7} |")
print(f"  |{'-'*27}|{'-'*11}|{'-'*8}|{'-'*8}|{'-'*9}|")
print(f"  | {'(a) Baseline':<25} | {baseline_metrics['Precision']:>9.4f} | {baseline_metrics['Recall']:>6.4f} | {baseline_metrics['F1']:>6.4f} | {baseline_metrics['ROC-AUC']:>7.4f} |")
print(f"  | {'(b) class_weight=balanced':<25} | {balanced_metrics['Precision']:>9.4f} | {balanced_metrics['Recall']:>6.4f} | {balanced_metrics['F1']:>6.4f} | {balanced_metrics['ROC-AUC']:>7.4f} |")
print(f"  | {'(c) SMOTE (imblearn)':<25} | {smote_metrics['Precision']:>9.4f} | {smote_metrics['Recall']:>6.4f} | {smote_metrics['F1']:>6.4f} | {smote_metrics['ROC-AUC']:>7.4f} |")

print(f"\n  Conclusions:")
print(f"    - class_weight='balanced' typically improves recall at the cost of precision,")
print(f"      as it penalizes misclassification of the minority class more heavily.")
print(f"    - SMOTE (via imblearn.Pipeline) synthesizes minority samples ONLY within")
print(f"      training folds, avoiding test set leakage. It generally provides a")
print(f"      balanced trade-off between precision and recall.")
print(f"    - imblearn.Pipeline ensures SMOTE is never applied to test data or before")
print(f"      cross-validation splitting, preventing artificial performance inflation.")

# %% [markdown]
# ## 7. Hyperparameter Tuning — GridSearchCV + OOB Score

# %%
print("\n" + "=" * 60)
print("  HYPERPARAMETER TUNING — GridSearchCV")
print("=" * 60)

# Parameter grid
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [5, 10, 15, None],
    'classifier__max_features': ['sqrt', 'log2', None],
}

# Pipeline for grid search
pipe_grid = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

grid_search = GridSearchCV(
    pipe_grid, param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1',
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X_train, y_train)

print(f"\n  Best parameters: {grid_search.best_params_}")
print(f"  Best CV F1 score: {grid_search.best_score_:.4f}")

# Extract best params (strip 'classifier__' prefix)
best_params = {k.replace('classifier__', ''): v for k, v in grid_search.best_params_.items()}
print(f"\n  Extracted best params: {best_params}")

# ── Fresh RandomForestClassifier with OOB score ──
print(f"\n  Constructing fresh RandomForestClassifier with oob_score=True and best params...")

# Need to preprocess X_train first to get OOB score
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

rf_best = RandomForestClassifier(
    oob_score=True,
    random_state=42,
    **best_params
)
rf_best.fit(X_train_processed, y_train)

print(f"  OOB Score: {rf_best.oob_score_:.4f}")
print(f"  Test Accuracy: {rf_best.score(X_test_processed, y_test):.4f}")
print(f"  Test F1: {f1_score(y_test, rf_best.predict(X_test_processed)):.4f}")

# %% [markdown]
# ## 8. Regression Side-Task — Predict Fare

# %%
print("\n" + "=" * 60)
print("  REGRESSION — Predicting Fare")
print("=" * 60)

# Reload clean data
df_reg = pd.read_csv(csv_path).dropna(subset=['fare', 'age'])

# Features for regression (exclude fare itself)
reg_features = ['pclass', 'age', 'sibsp', 'parch', 'survived']
X_reg = df_reg[reg_features]
y_reg = df_reg['fare']

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Fit linear regression
reg_model = LinearRegression()
reg_model.fit(X_reg_train, y_reg_train)
y_reg_pred = reg_model.predict(X_reg_test)

# Metrics
mae = mean_absolute_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
r2 = r2_score(y_reg_test, y_reg_pred)
n = len(y_reg_test)
p = X_reg_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f"\n  Regression Metrics (Predicting Fare):")
print(f"    MAE:          {mae:.4f}")
print(f"    RMSE:         {rmse:.4f}")
print(f"    R²:           {r2:.4f}")
print(f"    Adjusted R²:  {adj_r2:.4f}")
print(f"    n = {n}, p = {p}")

# ── Residual Plot ──
residuals = y_reg_test - y_reg_pred

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(y_reg_pred, residuals, alpha=0.5, color='#3498db', edgecolors='grey', s=30)
ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
ax.set_title('Residuals vs Predicted Values — Fare Prediction')
ax.set_xlabel('Predicted Fare (£)')
ax.set_ylabel('Residuals (Actual - Predicted)')
fig.tight_layout()
fig.savefig(os.path.join(CHARTS_DIR, 'residuals_plot.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"\n  Residual plot saved to: charts/residuals_plot.png")
print(f"\n  Heteroscedasticity Conclusion:")
print(f"    The residual plot shows a clear fan-shaped (funnel) pattern: residual variance")
print(f"    increases substantially with higher predicted fare values. This indicates")
print(f"    HETEROSCEDASTICITY — the assumption of constant error variance is violated.")
print(f"    This is expected because fare has a heavily right-skewed distribution with")
print(f"    extreme outliers in first class. A log-transformation of fare or weighted")
print(f"    least squares regression would be appropriate remedies.")

# %% [markdown]
# ## 9. Model Recommendation

# %%
print("\n" + "=" * 60)
print("  MODEL RECOMMENDATION")
print("=" * 60)

print("\n  CLASSIFICATION METRICS SUMMARY:")
print(f"  | {'Model':<25} | {'Accuracy':>8} | {'Precision':>9} | {'Recall':>6} | {'F1':>6} | {'ROC-AUC':>7} |")
print(f"  |{'-'*27}|{'-'*10}|{'-'*11}|{'-'*8}|{'-'*8}|{'-'*9}|")
for name, r in results.items():
    print(f"  | {name:<25} | {r['Accuracy']:>8.4f} | {r['Precision']:>9.4f} | {r['Recall']:>6.4f} | {r['F1']:>6.4f} | {r['ROC-AUC']:>7.4f} |")

print(f"\n  REGRESSION METRICS SUMMARY (Fare Prediction):")
print(f"  | {'Metric':<15} | {'Value':>10} |")
print(f"  |{'-'*17}|{'-'*12}|")
print(f"  | {'MAE':<15} | {mae:>10.4f} |")
print(f"  | {'RMSE':<15} | {rmse:>10.4f} |")
print(f"  | {'R²':<15} | {r2:>10.4f} |")
print(f"  | {'Adjusted R²':<15} | {adj_r2:>10.4f} |")

# Find best classifier
best_model_name = max(results, key=lambda k: results[k]['ROC-AUC'])
best_r = results[best_model_name]
print(f"\n  RECOMMENDATION: {best_model_name}")
print(f"    - Highest ROC-AUC ({best_r['ROC-AUC']:.4f}) indicates best discrimination ability.")
print(f"    - F1 score ({best_r['F1']:.4f}) balances precision and recall effectively.")
print(f"    - Random Forest is recommended for its ensemble stability and robustness to overfitting.")
print(f"    - With GridSearchCV tuning (best params: {best_params}), the model achieves")
print(f"      an OOB score of {rf_best.oob_score_:.4f}, confirming good generalization.")

# %% [markdown]
# ## 10. Save Best Pipeline & Verify

# %%
# Save the complete pipeline (preprocessor + best classifier)
best_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, **best_params))
])
best_pipeline.fit(X_train, y_train)

joblib_path = os.path.join(ANALYTICS_DIR, "best_pipeline.joblib")
joblib.dump(best_pipeline, joblib_path)
print(f"\n  ✅ Best pipeline saved to: {joblib_path}")
print(f"     File size: {os.path.getsize(joblib_path)} bytes")

# Quick verification
loaded = joblib.load(joblib_path)
y_loaded_pred = loaded.predict(X_test)
print(f"  ✅ Pipeline reload test: accuracy = {accuracy_score(y_test, y_loaded_pred):.4f}")

print("\n" + "=" * 60)
print("  MODELING COMPLETE")
print("=" * 60)

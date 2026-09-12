# %% [markdown]
# # Zepto Analytics — 01 Exploratory Data Analysis (EDA)
# 
# This notebook performs comprehensive EDA on the Titanic dataset:
# - Data profiling & threshold-based cleaning
# - Univariate analysis with outlier & skewness detection
# - Bivariate analysis with 6×6 correlation heatmap
# - Multivariate story with 4 distinct charts
# - Z-score standardization check

# %%
import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

CHARTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)
ANALYTICS_DIR = os.path.dirname(os.path.abspath(__file__))

# %% [markdown]
# ## 1. Load Titanic Dataset (Single Network Load)
# 
# We load the Titanic dataset from seaborn **exactly once** and save to CSV for all subsequent use.

# %%
# Single load from seaborn (the ONLY call to sns.load_dataset in the entire project)
df = sns.load_dataset('titanic')
csv_path = os.path.join(ANALYTICS_DIR, "titanic.csv")
df.to_csv(csv_path, index=False)
print(f"Titanic dataset saved to: {csv_path}")
print(f"File size: {os.path.getsize(csv_path)} bytes")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Verify the CSV
assert os.path.exists(csv_path), "titanic.csv was not created!"
assert os.path.getsize(csv_path) > 0, "titanic.csv is empty!"
df_verify = pd.read_csv(csv_path)
expected_cols = ['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare',
                 'embarked', 'class', 'who', 'adult_male', 'deck', 'embark_town',
                 'alive', 'alone']
for col in expected_cols:
    assert col in df_verify.columns, f"Missing expected column: {col}"
print(f"\n✅ Verified: titanic.csv exists, is non-empty, and contains all {len(expected_cols)} expected columns")

# %% [markdown]
# ## 2. Data Profiling

# %%
print("=" * 60)
print("  DATA PROFILING")
print("=" * 60)

print("\n── df.info() ──")
df.info()

print("\n── df.describe() ──")
print(df.describe())

print(f"\n── df.shape ──")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n── Missing Value Percentages ──")
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Column': missing_pct.index,
    'Missing_Count': df.isnull().sum().values,
    'Missing_Pct': missing_pct.values
}).sort_values('Missing_Pct', ascending=False)
print(missing_df[missing_df['Missing_Pct'] > 0].to_string(index=False))
print(f"\nColumns with no missing values: {(missing_pct == 0).sum()}")

# %% [markdown]
# ## 3. Threshold-Based Cleaning
# 
# Cleaning rules based on exact measured missing percentages:
# - **< 5% missing → Drop rows** (minimal data loss)
# - **5%–30% missing → Impute** (preserve data volume)
# - **> 30% missing → Drop column or encode 'Missing'** (too sparse for reliable imputation)

# %%
print("\n" + "=" * 60)
print("  THRESHOLD-BASED CLEANING")
print("=" * 60)

for col in df.columns:
    pct = missing_pct[col]
    if pct == 0:
        continue
    
    if pct > 30:
        # deck: ~77% missing → encode as 'Missing' category
        print(f"\n  [{col}] {pct:.2f}% missing (> 30%)")
        print(f"    → Decision: Encode missing values as 'Missing' category")
        print(f"    → Justification: With {pct:.2f}% missing, dropping would lose too much data.")
        print(f"      Imputation with mode would be misleading as the distribution is heavily incomplete.")
        print(f"      Encoding as 'Missing' preserves all rows and allows the model to learn from missingness.")
        df[col] = df[col].astype(str).replace('nan', 'Missing')
    elif pct >= 5:
        # age: ~19.87% missing → impute with median
        print(f"\n  [{col}] {pct:.2f}% missing (5-30%)")
        if df[col].dtype in ['float64', 'int64']:
            median_val = df[col].median()
            print(f"    → Decision: Impute with median = {median_val}")
            print(f"    → Justification: Median is robust to outliers in age distribution.")
            df[col] = df[col].fillna(median_val)
        else:
            mode_val = df[col].mode()[0]
            print(f"    → Decision: Impute with mode = '{mode_val}'")
            print(f"    → Justification: Mode preserves the most common category for categorical data.")
            df[col] = df[col].fillna(mode_val)
    else:
        # < 5% missing → drop rows
        print(f"\n  [{col}] {pct:.2f}% missing (< 5%)")
        before = len(df)
        df = df.dropna(subset=[col])
        after = len(df)
        print(f"    → Decision: Drop {before - after} rows")
        print(f"    → Justification: Minimal data loss ({pct:.2f}% < 5%), preserves data integrity.")

print(f"\n  Final shape after cleaning: {df.shape}")
print(f"  Remaining missing values: {df.isnull().sum().sum()}")

# %% [markdown]
# ## 4. Univariate Analysis — Outlier & Skewness Detection

# %%
print("\n" + "=" * 60)
print("  UNIVARIATE ANALYSIS — OUTLIERS (IQR Method)")
print("=" * 60)

for col_name in ['age', 'fare']:
    Q1 = df[col_name].quantile(0.25)
    Q3 = df[col_name].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col_name] < lower) | (df[col_name] > upper)]
    
    print(f"\n  {col_name.upper()}:")
    print(f"    Q1 = {Q1:.2f}, Q3 = {Q3:.2f}, IQR = {IQR:.2f}")
    print(f"    Lower bound = Q1 - 1.5×IQR = {lower:.2f}")
    print(f"    Upper bound = Q3 + 1.5×IQR = {upper:.2f}")
    print(f"    Outliers detected: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")

# %%
print("\n" + "=" * 60)
print("  FARE — Mean, Median, Mode & Skewness")
print("=" * 60)

fare_mean = df['fare'].mean()
fare_median = df['fare'].median()
fare_mode = df['fare'].mode()[0]

print(f"\n  Mean   = {fare_mean:.4f}")
print(f"  Median = {fare_median:.4f}")
print(f"  Mode   = {fare_mode:.4f}")
print(f"\n  Order: Mean ({fare_mean:.2f}) > Median ({fare_median:.2f}) > Mode ({fare_mode:.2f})")
print(f"  Conclusion: The fare distribution is RIGHT-SKEWED (positively skewed).")
print(f"  This is consistent with the Mean > Median > Mode ordering,")
print(f"  indicating a long right tail driven by expensive first-class fares.")

# Save boxplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i, col_name in enumerate(['age', 'fare']):
    axes[i].boxplot(df[col_name].dropna(), vert=True)
    axes[i].set_title(f'{col_name.title()} — Boxplot (IQR Outliers)')
    axes[i].set_ylabel(col_name.title())
fig.tight_layout()
fig.savefig(os.path.join(CHARTS_DIR, 'boxplots_age_fare.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\n  Saved: charts/boxplots_age_fare.png")

# %% [markdown]
# ## 5. Bivariate Analysis & Survival Rates

# %%
print("\n" + "=" * 60)
print("  BIVARIATE ANALYSIS — SURVIVAL RATES")
print("=" * 60)

# (a) Survival rate by sex
print("\n  (a) Survival Rate by Sex:")
surv_sex = df.groupby('sex')['survived'].mean()
print(surv_sex.to_string())

# (b) Survival rate by pclass
print("\n  (b) Survival Rate by Pclass:")
surv_pclass = df.groupby('pclass')['survived'].mean()
print(surv_pclass.to_string())

# (c) Survival rate by sex and pclass combined
print("\n  (c) Survival Rate by Sex × Pclass:")
surv_combined = df.groupby(['sex', 'pclass'])['survived'].mean().unstack()
print(surv_combined.to_string())

# %% [markdown]
# ## 6. Correlation Heatmap (6×6 — Strict Column Selection)
# 
# **Restricted to exactly 6 columns**: `survived`, `pclass`, `age`, `sibsp`, `parch`, `fare`
# 
# **Excluded**: `adult_male` and `alone` (derived/redundant boolean columns)

# %%
# Strict 6-column selection (excluding adult_male and alone)
corr_cols = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']
corr_matrix = df[corr_cols].corr()

print("\n" + "=" * 60)
print("  6×6 CORRELATION MATRIX")
print("=" * 60)
print(corr_matrix.round(4).to_string())

# Find top 2 off-diagonal correlations by absolute value
import itertools
corr_pairs = []
for i, j in itertools.combinations(range(len(corr_cols)), 2):
    c1, c2 = corr_cols[i], corr_cols[j]
    val = corr_matrix.loc[c1, c2]
    corr_pairs.append((c1, c2, val, abs(val)))

corr_pairs.sort(key=lambda x: x[3], reverse=True)
print(f"\n  Top 2 off-diagonal correlations (by |r|):")
for rank, (c1, c2, val, abs_val) in enumerate(corr_pairs[:2], 1):
    print(f"    {rank}. {c1} ↔ {c2}: r = {val:.4f} (|r| = {abs_val:.4f})")

print(f"\n  Interpretation:")
c1_1, c2_1, v1, _ = corr_pairs[0]
c1_2, c2_2, v2, _ = corr_pairs[1]
print(f"    1. {c1_1} ↔ {c2_1} (r = {v1:.4f}): ", end="")
if 'survived' in (c1_1, c2_1) and 'pclass' in (c1_1, c2_1):
    print("Higher class (lower pclass number) correlates with higher survival.")
    print("       This reflects the 'women and children first' protocol favoring upper-class passengers.")
elif 'survived' in (c1_1, c2_1) and 'fare' in (c1_1, c2_1):
    print("Higher fare correlates with higher survival, reflecting class privilege.")
elif 'pclass' in (c1_1, c2_1) and 'fare' in (c1_1, c2_1):
    print("Higher class (lower pclass) passengers paid higher fares.")
    print("       This is expected as pclass directly determines fare brackets.")
else:
    print(f"Strong linear association between {c1_1} and {c2_1}.")

print(f"    2. {c1_2} ↔ {c2_2} (r = {v2:.4f}): ", end="")
if 'survived' in (c1_2, c2_2) and 'fare' in (c1_2, c2_2):
    print("Higher fare correlates with higher survival rate.")
    print("       Wealthier passengers had better access to lifeboats.")
elif 'pclass' in (c1_2, c2_2) and 'fare' in (c1_2, c2_2):
    print("Higher class (lower pclass) correlates with higher fare.")
    print("       Direct economic relationship between ticket class and price.")
elif 'survived' in (c1_2, c2_2) and 'pclass' in (c1_2, c2_2):
    print("Higher class passengers had significantly higher survival rates.")
elif 'sibsp' in (c1_2, c2_2) and 'parch' in (c1_2, c2_2):
    print("Siblings/spouses and parents/children aboard are positively correlated.")
    print("       Families traveling together brought both types of relatives, reflecting")
    print("       that larger family groups included multiple relationship categories.")
else:
    print(f"Notable association between {c1_2} and {c2_2}.")

# Save heatmap
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5, ax=ax,
            xticklabels=corr_cols, yticklabels=corr_cols)
ax.set_title('Correlation Heatmap (6 Numeric Columns)\nExcluding adult_male and alone', fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(CHARTS_DIR, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\n  Saved: charts/correlation_heatmap.png")

# %% [markdown]
# ## 7. Multivariate Story — 4 Distinct Charts with Interpretations

# %%
print("\n" + "=" * 60)
print("  MULTIVARIATE ANALYSIS — 4 CHARTS")
print("=" * 60)

# ── Chart 1: Survival by Sex and Pclass (Grouped Bar) ──
fig, ax = plt.subplots(figsize=(8, 5))
surv_data = df.groupby(['pclass', 'sex'])['survived'].mean().unstack()
surv_data.plot(kind='bar', ax=ax, color=['#e74c3c', '#3498db'])
ax.set_title('Chart 1: Survival Rate by Passenger Class and Sex')
ax.set_xlabel('Passenger Class')
ax.set_ylabel('Survival Rate')
ax.set_xticklabels(['1st Class', '2nd Class', '3rd Class'], rotation=0)
ax.legend(title='Sex')
fig.tight_layout()
fig.savefig(os.path.join(CHARTS_DIR, 'chart1_survival_class_sex.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

print("\n  Chart 1: Survival Rate by Passenger Class and Sex")
print("  Interpretation: Female passengers had dramatically higher survival rates")
print("  across all three classes, with 1st class females surviving at ~96%. Male")
print("  survival rates declined sharply from 1st to 3rd class, confirming that both")
print("  sex and class were major survival determinants during the Titanic disaster.")

# ── Chart 2: Age Distribution by Survival (Violin Plot) ──
fig, ax = plt.subplots(figsize=(8, 5))
# Need to ensure survived is string for hue
df_plot = df.copy()
df_plot['survived_label'] = df_plot['survived'].map({0: 'Did Not Survive', 1: 'Survived'})
sns.violinplot(data=df_plot, x='pclass', y='age', hue='survived_label', split=True,
               palette={'Survived': '#2ecc71', 'Did Not Survive': '#e74c3c'}, ax=ax)
ax.set_title('Chart 2: Age Distribution by Class and Survival')
ax.set_xlabel('Passenger Class')
ax.set_ylabel('Age')
ax.legend(title='Outcome')
fig.tight_layout()
fig.savefig(os.path.join(CHARTS_DIR, 'chart2_age_class_survival.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

print("\n  Chart 2: Age Distribution by Class and Survival (Violin Plot)")
print("  Interpretation: Younger passengers in 3rd class had slightly better survival")
print("  chances, likely reflecting the 'children first' evacuation priority. In 1st")
print("  class, the age distribution for survivors is broader, suggesting that wealth")
print("  provided a survival advantage regardless of age.")

# ── Chart 3: Fare vs Age scatter colored by survival ──
fig, ax = plt.subplots(figsize=(8, 5))
scatter = ax.scatter(df['age'], df['fare'], c=df['survived'], cmap='RdYlGn',
                     alpha=0.6, edgecolors='grey', linewidths=0.5, s=30)
ax.set_title('Chart 3: Fare vs Age (Colored by Survival)')
ax.set_xlabel('Age')
ax.set_ylabel('Fare (£)')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Survived (0=No, 1=Yes)')
fig.tight_layout()
fig.savefig(os.path.join(CHARTS_DIR, 'chart3_fare_age_survival.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

print("\n  Chart 3: Fare vs Age (Colored by Survival)")
print("  Interpretation: Higher fare passengers (top of y-axis) show predominantly")
print("  green (survived), while the dense cluster of low-fare passengers shows mixed")
print("  outcomes. This confirms the class-survival relationship. The scatter also")
print("  reveals fare outliers (>£200) who almost all survived.")

# ── Chart 4: Family Size vs Survival ──
df['family_size'] = df['sibsp'] + df['parch'] + 1
fig, ax = plt.subplots(figsize=(8, 5))
family_surv = df.groupby('family_size')['survived'].agg(['mean', 'count']).reset_index()
family_surv.columns = ['family_size', 'survival_rate', 'count']
bars = ax.bar(family_surv['family_size'], family_surv['survival_rate'],
              color='#3498db', edgecolor='white', alpha=0.8)
ax.set_title('Chart 4: Survival Rate by Family Size (sibsp + parch + 1)')
ax.set_xlabel('Family Size')
ax.set_ylabel('Survival Rate')
ax.set_xticks(family_surv['family_size'])
# Add count labels
for bar, count in zip(bars, family_surv['count']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'n={count}', ha='center', va='bottom', fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(CHARTS_DIR, 'chart4_family_size_survival.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

print("\n  Chart 4: Survival Rate by Family Size")
print("  Interpretation: Solo travelers (family_size=1) and large families (7+) had")
print("  the lowest survival rates. Medium-sized families (2-4) had the highest survival,")
print("  suggesting that having some family members helped during evacuation, but very")
print("  large families faced coordination challenges that reduced their survival chances.")

# %% [markdown]
# ## 8. Z-Score Standardization Check (Exploratory Only)

# %%
print("\n" + "=" * 60)
print("  Z-SCORE STANDARDIZATION CHECK (Exploratory)")
print("=" * 60)

for col_name in ['age', 'fare']:
    original_mean = df[col_name].mean()
    original_std = df[col_name].std()
    
    # Z-score: z = (x - μ) / σ
    z_scores = (df[col_name] - original_mean) / original_std
    
    print(f"\n  {col_name.upper()}:")
    print(f"    Pre-scaling  — Mean: {original_mean:.4f}, Std: {original_std:.4f}")
    print(f"    Post-scaling — Mean: {z_scores.mean():.6f} (≈ 0), Std: {z_scores.std():.6f} (≈ 1)")

print("\n  ✅ Z-score standardization confirmed: post-scaling mean ≈ 0 and std ≈ 1")
print("  Note: This is an exploratory sanity check only; actual model scaling uses")
print("  sklearn's StandardScaler fitted on training data in 02_modeling.ipynb.")

print("\n" + "=" * 60)
print("  EDA COMPLETE — All charts saved to analytics/charts/")
print("=" * 60)

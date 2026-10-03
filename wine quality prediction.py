# %% [markdown]
# # Task 2 · Wine Quality Prediction
# Compare Random Forest, SGD and SVC on the UCI Wine Quality dataset.
#
# In VS Code: install the Python + Jupyter extensions, then click "Run Cell"
# above each `# %%` block (or save as .ipynb via "Export as Jupyter Notebook").
# pip install pandas numpy scikit-learn seaborn matplotlib

# %% 1. Imports
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns   

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             f1_score)

sns.set_theme(style="whitegrid")
RANDOM_STATE = 42

# %% 2. Load dataset
# Option A (default): download directly from UCI.
# Option B: download the CSVs from Kaggle/UCI and set the local paths below.
RED_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
WHITE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
RED_PATH, WHITE_PATH = RED_URL, WHITE_URL   # e.g. "winequality-red.csv", "winequality-white.csv"

red = pd.read_csv(RED_PATH, sep=";")
white = pd.read_csv(WHITE_PATH, sep=";")
red["wine_type"] = 0     # 0 = red
white["wine_type"] = 1   # 1 = white
df = pd.concat([red, white], ignore_index=True)

print("Shape:", df.shape)
display(df.head()) if "display" in globals() else print(df.head())
print("\nInfo:")
df.info()
print("\nMissing values:\n", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())
print("\nSummary statistics:")
print(df.describe().T)

# %% 3. Class distribution of quality scores
counts = df["quality"].value_counts().sort_index()
print(counts)
print((counts / len(df) * 100).round(2).astype(str) + " %")

plt.figure(figsize=(7, 4))
sns.countplot(x="quality", data=df, palette="viridis")
plt.title("Distribution of Wine Quality Scores")
plt.show()

# %% 4. EDA: distribution of every chemical feature
features = [c for c in df.columns if c not in ("quality", "wine_type")]

fig, axes = plt.subplots(4, 3, figsize=(16, 14))
for ax, col in zip(axes.ravel(), features):
    sns.histplot(df[col], kde=True, ax=ax, color="steelblue")
    ax.set_title(col)
for ax in axes.ravel()[len(features):]:
    ax.axis("off")
plt.tight_layout()
plt.show()

# Box plots of each feature vs quality (shows which features separate quality)
fig, axes = plt.subplots(4, 3, figsize=(16, 14))
for ax, col in zip(axes.ravel(), features):
    sns.boxplot(x="quality", y=col, data=df, ax=ax, palette="viridis")
    ax.set_title(f"{col} vs quality")
for ax in axes.ravel()[len(features):]:
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% 5. Correlation heatmap
plt.figure(figsize=(12, 9))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

print("Correlation with quality:")
print(corr["quality"].drop("quality").sort_values(ascending=False))

# %% [markdown]
# ## Class imbalance discussion
# - Scores 5 and 6 make up the large majority of wines; the extremes
#   (3, 4, 8, 9) have very few samples (score 9 has only ~5 rows).
# - Impact on modelling: accuracy becomes misleading (a model that always
#   predicts "6" already scores ~44%), models become biased toward majority
#   classes, and rare classes get near-zero recall. A stratified split can't
#   be done for classes with <2 samples.
# - Mitigation used here: (1) bin quality into fewer, better-populated groups,
#   (2) stratified splitting, (3) `class_weight="balanced"`,
#   (4) report macro-F1 and per-class recall, not just accuracy.

# %% 6. Feature engineering: bin quality scores
# Choose the grouping: "3class" (low/medium/high) or "binary" (good/bad)
BINNING = "3class"

if BINNING == "3class":
    # low: <=5, medium: 6, high: >=7
    df["target"] = pd.cut(df["quality"], bins=[0, 5, 6, 10],
                          labels=["low", "medium", "high"])
    # Justification: keeps ordinal information, groups the sparse extreme
    # scores into populated classes, and mirrors a practical grading
    # (reject / standard / premium).
else:
    # good: >=7, bad: <=6
    df["target"] = pd.cut(df["quality"], bins=[0, 6, 10], labels=["bad", "good"])
    # Justification: simplest decision for deployment (accept/reject), but
    # more imbalanced (~20% good).

class_names = list(df["target"].cat.categories)
print(df["target"].value_counts().reindex(class_names))
print((df["target"].value_counts(normalize=True).reindex(class_names) * 100).round(2))

sns.countplot(x="target", data=df, order=class_names, palette="viridis")
plt.title(f"Target distribution after binning ({BINNING})")
plt.show()

# %% 7. Stratified train/test split
X = df.drop(columns=["quality", "target"])
y = df["target"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print("Train:", X_train.shape, "Test:", X_test.shape)
print("\nTrain ratios:\n", y_train.value_counts(normalize=True).round(3))
print("\nTest ratios:\n", y_test.value_counts(normalize=True).round(3))

# %% 8. Define models (SGD and SVC need scaled features; RF does not)
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1),
    "SGD": make_pipeline(
        StandardScaler(),
        SGDClassifier(loss="hinge", penalty="l2", alpha=1e-4,
                      max_iter=2000, tol=1e-3, class_weight="balanced",
                      random_state=RANDOM_STATE)),
    "SVC": make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
            random_state=RANDOM_STATE)),
}

# %% 9. Train and evaluate each model
labels = sorted(y.unique())
if BINNING == "3class":
    labels = ["low", "medium", "high"]

results = []
fitted = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    fitted[name] = model
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    results.append({"Model": name, "Accuracy": acc,
                    "Macro F1": macro_f1, "Weighted F1": weighted_f1})

    print("=" * 60)
    print(f"{name}  |  accuracy = {acc:.4f}")
    print("=" * 60)
    print(classification_report(y_test, y_pred, labels=labels, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix · {name}")
    plt.grid(False)
    plt.show()

# %% 10. Random Forest feature importance
rf = fitted["Random Forest"]
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values()

plt.figure(figsize=(8, 6))
importances.plot(kind="barh", color="teal")
plt.title("Random Forest · Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()
print(importances.sort_values(ascending=False).round(4))

# %% 11. Comparison table
comparison = (pd.DataFrame(results).set_index("Model").round(4)
              .sort_values("Macro F1", ascending=False))
print(comparison)

comparison.plot(kind="bar", figsize=(8, 5), colormap="viridis")
plt.title("Model Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.show()

# %% 12. Conclusion (auto-picks the best model, then explains)
best = comparison.index[0]
print(f"Best model by macro-F1: {best}")
print(comparison.loc[best])

# %% [markdown]
# ## Conclusion
# - **Random Forest** is typically the strongest here: it captures non-linear
#   interactions between features (alcohol, volatile acidity, density,
#   sulphates), is robust to outliers and unscaled features, handles
#   imbalance with `class_weight`, and gives interpretable feature
#   importances. Cost: larger model and slower inference than a linear model.
# - **SVC (RBF)** is usually close behind but is sensitive to scaling and
#   hyperparameters (C, gamma), and training time grows quickly with data size.
# - **SGD (linear)** is fastest and lightest but a linear boundary underfits
#   this non-linear problem, giving the lowest scores.
# - **Recommendation:** deploy Random Forest (confirm with the table above),
#   after tuning with `GridSearchCV`/cross-validation. Judge by macro-F1 and
#   per-class recall rather than accuracy alone, because of class imbalance.
#   Note the rare "high" class remains the hardest to predict.
#
# Possible improvements: SMOTE oversampling, cross-validation, hyperparameter
# tuning, gradient boosting (XGBoost/LightGBM).
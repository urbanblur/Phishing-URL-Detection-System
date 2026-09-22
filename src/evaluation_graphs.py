import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# LOAD DATASET
# ============================================================

DATASET_PATH = "dataset/phishing_urls.csv"

df = pd.read_csv(DATASET_PATH)

X_text = df["URL"].astype(str)
y = df["label"]
groups = df["Domain"].astype(str)


# ============================================================
# DOMAIN-DISJOINT TEST SPLIT
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X_text, y, groups=groups)
)

X_test = X_text.iloc[test_idx]
y_test = y.iloc[test_idx]


# ============================================================
# LOAD DOMAIN-AWARE MODEL
# ============================================================

saved_model = joblib.load(
    "models/domain_aware_model.pkl"
)

model = saved_model["model"]
vectorizer = saved_model["vectorizer"]


# ============================================================
# TRANSFORM TEST DATA
# ============================================================

X_test_vectorized = vectorizer.transform(X_test)

y_pred = model.predict(X_test_vectorized)


# ============================================================
# CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=0)
recall = recall_score(y_test, y_pred, pos_label=0)
f1 = f1_score(y_test, y_pred, pos_label=0)

print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1 Score : {f1 * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

import os

os.makedirs("evaluation", exist_ok=True)


# ============================================================
# GRAPH 1 — PERFORMANCE METRICS
# ============================================================

metrics = {
    "Accuracy": accuracy * 100,
    "Precision": precision * 100,
    "Recall": recall * 100,
    "F1 Score": f1 * 100
}

plt.figure(figsize=(9, 6))

plt.bar(
    metrics.keys(),
    metrics.values()
)

plt.ylim(0, 105)

plt.ylabel("Percentage")

plt.title(
    "Domain-Aware Model Performance"
)

for i, value in enumerate(metrics.values()):
    plt.text(
        i,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "evaluation/model_performance.png",
    dpi=300
)

plt.show()


# ============================================================
# GRAPH 2 — CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Phishing",
        "Legitimate"
    ]
)

fig, ax = plt.subplots(
    figsize=(8, 6)
)

disp.plot(
    ax=ax
)

plt.title(
    "Domain-Aware Model Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "evaluation/confusion_matrix.png",
    dpi=300
)

plt.show()


print("\nGraphs saved successfully.")
print("evaluation/model_performance.png")
print("evaluation/confusion_matrix.png")
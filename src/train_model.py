import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 1. LOAD DATASET
# ============================================================

DATASET_PATH = "dataset/phishing_urls.csv"

print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")


# ============================================================
# 2. PREPARE FEATURES
# ============================================================

# Target column
target = "label"

# Columns that should NOT be used directly as ML features
columns_to_exclude = [
    "label",
    "URL",
    "FILENAME",
    "Domain",
    "TLD"
]

# Keep only numeric columns
numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

# Remove excluded columns if present
feature_columns = [
    column for column in numeric_columns
    if column not in columns_to_exclude
]

X = df[feature_columns]
y = df[target]

print("\nNumber of features:", len(feature_columns))
print("\nFeatures being used:")
for feature in feature_columns:
    print("-", feature)


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 4. TRAIN RANDOM FOREST MODEL
# ============================================================

print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("Model training completed.")


# ============================================================
# 5. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 6. EVALUATE MODEL
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ============================================================
# 7. SAVE MODEL
# ============================================================

MODEL_PATH = "models/phishing_url_model.pkl"

joblib.dump(
    {
        "model": model,
        "features": feature_columns
    },
    MODEL_PATH
)

print("\nModel saved successfully:")
print(MODEL_PATH)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
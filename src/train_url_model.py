import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# SETTINGS
# ============================================================

DATASET_PATH = "dataset/phishing_urls.csv"
MODEL_PATH = "models/url_text_model.pkl"


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("PHISHING & MALICIOUS URL DETECTION")
print("URL TEXT MODEL TRAINING")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

df = df[["URL", "label"]].dropna()

print("Dataset loaded.")
print("Rows:", len(df))


# ============================================================
# PREPARE DATA
# ============================================================

X = df["URL"].astype(str)

# PhiUSIIL:
# 0 = Phishing
# 1 = Legitimate
y = df["label"].astype(int)

print("\nLabel distribution:")
print(y.value_counts())


# ============================================================
# TRAIN / TEST SPLIT
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
# TF-IDF FEATURE EXTRACTION
# ============================================================

print("\nCreating character-level TF-IDF features...")

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_features=100000,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF features created.")
print("Feature count:", len(vectorizer.get_feature_names_out()))


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_tfidf, y_train)

print("Model training complete.")


# ============================================================
# PREDICTION
# ============================================================

print("\nEvaluating model...")

y_pred = model.predict(X_test_tfidf)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Phishing",
            "Legitimate"
        ]
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ============================================================
# SAVE MODEL
# ============================================================

saved_model = {
    "model": model,
    "vectorizer": vectorizer
}

joblib.dump(saved_model, MODEL_PATH)

print("\n" + "=" * 60)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 60)

print("\nSaved to:")
print(MODEL_PATH)
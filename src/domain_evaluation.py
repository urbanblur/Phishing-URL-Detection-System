import pandas as pd
import tldextract
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


DATASET_PATH = "dataset/phishing_urls.csv"


print("=" * 60)
print("DOMAIN-AWARE MODEL EVALUATION")
print("=" * 60)


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH)

df = df[["URL", "label"]].dropna()

df["URL"] = df["URL"].astype(str)


# ============================================================
# EXTRACT REGISTERED DOMAIN
# ============================================================

def get_domain(url):

    extracted = tldextract.extract(url)

    if extracted.domain and extracted.suffix:
        return extracted.domain + "." + extracted.suffix

    return extracted.domain


df["domain"] = df["URL"].apply(get_domain)

print("\nTotal URLs:", len(df))
print("Unique domains:", df["domain"].nunique())


# ============================================================
# DOMAIN-AWARE TRAIN / TEST SPLIT
# ============================================================

X = df["URL"]
y = df["label"]
groups = df["domain"]


splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_index, test_index = next(
    splitter.split(X, y, groups=groups)
)


X_train = X.iloc[train_index]
X_test = X.iloc[test_index]

y_train = y.iloc[train_index]
y_test = y.iloc[test_index]


train_domains = set(groups.iloc[train_index])
test_domains = set(groups.iloc[test_index])


print("\nTraining URLs:", len(X_train))
print("Testing URLs:", len(X_test))

print("\nTraining domains:", len(train_domains))
print("Testing domains:", len(test_domains))

print(
    "Domain overlap:",
    len(train_domains.intersection(test_domains))
)


# ============================================================
# TF-IDF
# ============================================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_features=100000,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# ============================================================
# TRAIN MODEL
# ============================================================

print("Training model...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_tfidf, y_train)


# ============================================================
# EVALUATION
# ============================================================

y_pred = model.predict(X_test_tfidf)


accuracy = accuracy_score(y_test, y_pred)


print("\n" + "=" * 60)
print("DOMAIN-AWARE PERFORMANCE")
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

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)
# ============================================================
# SAVE DOMAIN-AWARE MODEL
# ============================================================

import joblib
import os

os.makedirs("models", exist_ok=True)

saved_model = {
    "model": model,
    "vectorizer": vectorizer
}

joblib.dump(
    saved_model,
    "models/domain_aware_model.pkl"
)

print("\nDomain-aware model saved successfully.")
print("Saved to: models/domain_aware_model.pkl")
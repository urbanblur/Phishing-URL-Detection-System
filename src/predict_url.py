import joblib


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "models/url_text_model.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("PHISHING & MALICIOUS URL DETECTION SYSTEM")
print("=" * 60)

print("\nLoading model...")

saved_model = joblib.load(MODEL_PATH)

model = saved_model["model"]
vectorizer = saved_model["vectorizer"]

print("Model loaded successfully.")


# ============================================================
# GET URL
# ============================================================

url = input("\nEnter URL to check: ").strip()

if not url:
    print("No URL entered.")
    raise SystemExit


# ============================================================
# CONVERT URL INTO TF-IDF FEATURES
# ============================================================

url_features = vectorizer.transform([url])


# ============================================================
# PREDICTION
# ============================================================

prediction = model.predict(url_features)[0]

probabilities = model.predict_proba(url_features)[0]

phishing_probability = probabilities[0] * 100
legitimate_probability = probabilities[1] * 100


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n" + "=" * 60)

print("URL:", url)

if prediction == 0:

    print("RESULT: PHISHING / MALICIOUS URL")
    print(f"Model confidence: {phishing_probability:.2f}%")

else:

    print("RESULT: LEGITIMATE URL")
    print(f"Model confidence: {legitimate_probability:.2f}%")

print("=" * 60)
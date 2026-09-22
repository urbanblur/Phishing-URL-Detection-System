from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# ============================================================
# LOAD DOMAIN-AWARE MODEL
# ============================================================

MODEL_PATH = "models/domain_aware_model.pkl"

saved_model = joblib.load(MODEL_PATH)

model = saved_model["model"]
vectorizer = saved_model["vectorizer"]


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    confidence = None
    url = None

    if request.method == "POST":

        url = request.form.get("url", "").strip()

        if url:

            # Convert URL into TF-IDF features
            features = vectorizer.transform([url])

            # Prediction
            prediction = model.predict(features)[0]

            # Confidence
            probabilities = model.predict_proba(features)[0]

            phishing_probability = probabilities[0] * 100
            legitimate_probability = probabilities[1] * 100

            # Dataset mapping:
            # 0 = Phishing
            # 1 = Legitimate

            if prediction == 0:
                result = "PHISHING / MALICIOUS URL"
                confidence = phishing_probability
            else:
                result = "LEGITIMATE URL"
                confidence = legitimate_probability

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        url=url
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)
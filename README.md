# Phishing and Malicious URL Detection System

A machine learning-based web application that analyzes URLs and classifies them as either phishing or legitimate.

## Project Overview

Phishing attacks use fraudulent URLs to trick users into visiting malicious websites or providing sensitive information.

This project develops a machine learning system that analyzes the textual characteristics of a URL and predicts whether it is:

- Phishing / Malicious
- Legitimate

The system provides a web interface built using Flask.

---

## Technologies Used

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- HTML
- CSS

---

## Machine Learning Approach

The system uses:

### Feature Representation

Character-level TF-IDF (Term Frequency-Inverse Document Frequency)

The URL is converted into numerical features based on character patterns.

### Classification Algorithm

Logistic Regression

The classifier uses the extracted TF-IDF representation to classify URLs.

---

## Dataset

The project uses the PhiUSIIL Phishing URL Dataset.

The dataset contains labeled URLs belonging to two classes:

- `0` → Phishing
- `1` → Legitimate

The dataset contains approximately 235,795 URLs.

---

## Model Evaluation

The model was evaluated using a domain-disjoint train/test split.

This prevents URLs from the same domain from appearing in both the training and testing sets.

### Results

| Metric | Result |
|---|---:|
| Accuracy | 99.70% |
| Precision | 100.00% |
| Recall | 99.29% |
| F1 Score | 99.65% |

### Confusion Matrix

```text
                 Predicted
                 Phishing  Legitimate

Actual Phishing     19658       140
Actual Legitimate      0      26923
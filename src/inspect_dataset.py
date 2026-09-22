import pandas as pd

# Load dataset
file_path = "dataset/phishing_urls.csv"

df = pd.read_csv(file_path)

print("=" * 60)
print("PHISHING URL DATASET INSPECTION")
print("=" * 60)

# Basic information
print("\nDataset shape:")
print(df.shape)

print("\nNumber of rows:")
print(len(df))

print("\nNumber of columns:")
print(len(df.columns))

# Column names
print("\nColumns:")
for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")

# Label distribution
print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nLabel percentages:")
print(df["label"].value_counts(normalize=True) * 100)

# Missing values
print("\nMissing values:")
missing = df.isnull().sum()
print(missing[missing > 0])

print("\nFirst 5 rows:")
print(df.head())

print("\n" + "=" * 60)
print("INSPECTION COMPLETE")
print("=" * 60)
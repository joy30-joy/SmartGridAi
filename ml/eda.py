
import pandas as pd

# Load dataset
df = pd.read_csv("data/texas_outages_synthetic_50000-5.csv")

# Basic information
print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- DATASET SHAPE ---")
print(df.shape)

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())

print("\n--- STATISTICAL SUMMARY ---")
print(df.describe(include="all"))
print("\n--- ALL COLUMNS ---")
print(df.columns.tolist())

print("\n--- TARGET CANDIDATES ---")
print(df[["duration", "mean_customers", "zone_usage"]].describe())

print("\n--- ERCOT ZONE DISTRIBUTION ---")
print(df["ERCOT_zone"].value_counts())

print("\n--- DATA SAMPLE ---")
print(df.sample(5, random_state=42).to_string(index=False))

print("\n--- DATE RANGE ---")
df["date"] = pd.to_datetime(df["date"])
print("Start:", df["date"].min())
print("End:", df["date"].max())
print("Unique dates:", df["date"].nunique())

print("\n--- ELECTRICITY USAGE DISTRIBUTION ---")
print(df["zone_usage"].describe(
    percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
))

print("\n--- USAGE BY ERCOT ZONE ---")
print(df.groupby("ERCOT_zone")["zone_usage"].agg(
    ["count", "mean", "min", "max"]
))

print("\n--- CORRELATION WITH USAGE ---")
print(
    df.select_dtypes(include="number")
      .corr()["zone_usage"]
      .sort_values(ascending=False)
)

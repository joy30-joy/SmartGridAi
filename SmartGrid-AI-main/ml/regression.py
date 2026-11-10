
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# -----------------------------
# 1. LOAD DATASET
# -----------------------------

DATA_PATH = "data/texas_synthetic_heat_sensitive_50000.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded:", df.shape)

# -----------------------------
# 2. FEATURE ENGINEERING
# -----------------------------

df["date"] = pd.to_datetime(df["date"])

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["dayofyear"] = df["date"].dt.dayofyear

# Cooling Degree Days
df["CDD"] = np.maximum(0, df["temperature_mean"] - 18.3)

# Heating Degree Days
df["HDD"] = np.maximum(0, 18.3 - df["temperature_mean"])

# -----------------------------
# 3. SELECT FEATURES
# -----------------------------

numeric_features = [
    "latitude",
    "longitude",
    "temperature_mean",
    "temperature_max",
    "temperature_min",
    "precipitation_mm",
    "wind_speed_mps",
    "snowfall_cm",
    "humidity",
    "year",
    "month",
    "dayofyear",
    "CDD",
    "HDD"
]

categorical_features = ["ERCOT_zone"]

X = df[numeric_features + categorical_features]
y = df["zone_usage"]

# -----------------------------
# 4. CHRONOLOGICAL TRAIN-TEST SPLIT
# -----------------------------

df_sorted = df.sort_values("date").reset_index(drop=True)

X = df_sorted[numeric_features + categorical_features]
y = df_sorted["zone_usage"]

split_index = int(len(df_sorted) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# -----------------------------
# 5. PREPROCESSING
# -----------------------------

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# -----------------------------
# 6. MODELS
# -----------------------------

models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),

    "XGBoost": XGBRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        objective="reg:squarederror",
        n_jobs=-1
    )
}

# -----------------------------
# 7. TRAIN AND EVALUATE
# -----------------------------

results = []
trained_models = {}

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    print(f"\nTraining {name}...")

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    trained_models[name] = pipeline

# -----------------------------
# 8. COMPARE MODELS
# -----------------------------

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("RMSE")

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

print("\nBest model:", best_model_name)

# -----------------------------
# 9. SAVE BEST MODEL
# -----------------------------

os.makedirs("artifacts", exist_ok=True)

joblib.dump(best_model, "artifacts/usage_regression_model.pkl")

print("\nBest regression model saved successfully!")

# -----------------------------
# 10. TEMPERATURE SENSITIVITY TEST
# -----------------------------

sample = X_test.iloc[[0]].copy()

print("\n========== TEMPERATURE SENSITIVITY ==========")

for temp in [25, 30, 35, 40]:

    test_sample = sample.copy()

    test_sample["temperature_mean"] = temp
    test_sample["temperature_max"] = temp + 5
    test_sample["temperature_min"] = temp - 5

    test_sample["CDD"] = max(0, temp - 18.3)
    test_sample["HDD"] = max(0, 18.3 - temp)

    predicted_usage = best_model.predict(test_sample)[0]

    print(f"Temperature: {temp}°C | Predicted usage: {predicted_usage:,.2f}")

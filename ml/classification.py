
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# --------------------------------
# 1. LOAD DATASET
# --------------------------------

DATA_PATH = "data/texas_synthetic_heat_sensitive_50000.csv"

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])

print("Dataset loaded:", df.shape)

# --------------------------------
# 2. FEATURE ENGINEERING
# --------------------------------

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["dayofyear"] = df["date"].dt.dayofyear

df["CDD"] = np.maximum(0, df["temperature_mean"] - 18.3)
df["HDD"] = np.maximum(0, 18.3 - df["temperature_mean"])

# --------------------------------
# 3. DEFINE FEATURES AND TARGET
# --------------------------------

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

feature_columns = numeric_features + categorical_features

# Define high-usage threshold using training data only.
# We first sort chronologically to avoid using future data.

df = df.sort_values("date").reset_index(drop=True)

split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

usage_threshold = train_df["zone_usage"].quantile(0.75)

train_df["grid_stress"] = (
    train_df["zone_usage"] >= usage_threshold
).astype(int)

test_df["grid_stress"] = (
    test_df["zone_usage"] >= usage_threshold
).astype(int)

X_train = train_df[feature_columns]
X_test = test_df[feature_columns]

y_train = train_df["grid_stress"]
y_test = test_df["grid_stress"]

print("\nUsage threshold:", round(usage_threshold, 2))
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts())

# --------------------------------
# 4. PREPROCESSING
# --------------------------------

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# --------------------------------
# 5. CLASSIFICATION MODELS
# --------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ),

    "XGBoost": XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )
}

# --------------------------------
# 6. TRAIN AND EVALUATE
# --------------------------------

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

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    trained_models[name] = pipeline

# --------------------------------
# 7. COMPARE MODELS
# --------------------------------

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n========== CLASSIFICATION RESULTS ==========")
print(results_df.to_string(index=False))

# --------------------------------
# 8. SELECT BEST MODEL
# --------------------------------

best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

print("\nBest classification model:", best_model_name)

# --------------------------------
# 9. DETAILED EVALUATION
# --------------------------------

final_predictions = best_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(
    y_test,
    final_predictions,
    target_names=["Normal Usage", "High Usage"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, final_predictions))

# --------------------------------
# 10. SAVE MODEL
# --------------------------------

os.makedirs("artifacts", exist_ok=True)

model_artifact = {
    "model": best_model,
    "threshold": usage_threshold,
    "feature_columns": feature_columns,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features
}

joblib.dump(
    model_artifact,
    "artifacts/usage_classification_model.pkl"
)

print("\nBest classification model saved successfully!")

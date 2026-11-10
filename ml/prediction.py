
import os
import joblib
import numpy as np
import pandas as pd

# --------------------------------
# 1. LOAD TRAINED MODELS
# --------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REG_PATH = os.path.join(BASE_DIR, "artifacts", "usage_regression_model.pkl")
if not os.path.exists(REG_PATH):
    REG_PATH = "artifacts/usage_regression_model.pkl"

CLS_PATH = os.path.join(BASE_DIR, "artifacts", "usage_classification_model.pkl")
if not os.path.exists(CLS_PATH):
    CLS_PATH = "artifacts/usage_classification_model.pkl"

regression_model = joblib.load(REG_PATH)
classification_artifact = joblib.load(CLS_PATH)

classification_model = classification_artifact["model"]
usage_threshold = classification_artifact["threshold"]

print("Regression and classification models loaded!")


# --------------------------------
# 2. PREPARE INPUT FEATURES
# --------------------------------

def prepare_input(input_data):

    df = pd.DataFrame([input_data])

    df["date"] = pd.to_datetime(df["date"])

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["dayofyear"] = df["date"].dt.day_of_year

    df["CDD"] = np.maximum(
        0, df["temperature_mean"] - 18.3
    )

    df["HDD"] = np.maximum(
        0, 18.3 - df["temperature_mean"]
    )

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

    return df[feature_columns]


# --------------------------------
# 3. RUN COMPLETE PREDICTION
# --------------------------------

def predict_grid_status(input_data):

    X = prepare_input(input_data)

    # Regression prediction
    predicted_usage = float(
        regression_model.predict(X)[0]
    )

    # Classification prediction
    risk_status = int(
        classification_model.predict(X)[0]
    )

    # Probability of high usage
    probabilities = classification_model.predict_proba(X)[0]

    high_usage_probability = float(probabilities[1]) * 100

    # Create query for RAG
    if risk_status == 1:
        status_text = "High Usage Risk"
    else:
        status_text = "Normal Usage"

    rag_query = f"""
    Explain the electricity usage prediction.

    Predicted electricity usage: {predicted_usage:.2f}
    Usage threshold: {usage_threshold:.2f}
    Risk status: {status_text}
    High usage probability: {high_usage_probability:.2f}%

    Explain possible reasons for this prediction,
    potential electricity grid concerns, and
    recommendations using the provided documents.
    """

    return {
        "predicted_usage": round(predicted_usage, 2),
        "risk_status": status_text,
        "risk_probability": round(high_usage_probability, 2),
        "usage_threshold": round(float(usage_threshold), 2),
        "rag_query": rag_query
    }


# --------------------------------
# 4. TEST PREDICTION
# --------------------------------

if __name__ == "__main__":

    sample_input = {
        "latitude": 32.7767,
        "longitude": -96.7970,
        "ERCOT_zone": "NORTH",
        "date": "2026-07-15",
        "temperature_mean": 38,
        "temperature_max": 42,
        "temperature_min": 34,
        "precipitation_mm": 0,
        "wind_speed_mps": 4,
        "snowfall_cm": 0,
        "humidity": 55
    }

    result = predict_grid_status(sample_input)

    print("\n========== SMARTGRID PREDICTION ==========")

    for key, value in result.items():
        if key != "rag_query":
            print(f"{key}: {value}")

    print("\nRAG Query:")
    print(result["rag_query"])

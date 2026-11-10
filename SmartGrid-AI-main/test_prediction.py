
from ml.prediction import predict_grid_status

sample_input = {
    "latitude": 32.75,
    "longitude": -96.77,
    "ERCOT_zone": "NORTH",
    "date": "2026-11-10",
    "temperature_mean": 25.0,
    "temperature_max": 32.0,
    "temperature_min": 18.0,
    "precipitation_mm": 0.0,
    "wind_speed_mps": 5.0,
    "snowfall_cm": 0.0,
    "humidity": 60.0
}

result = predict_grid_status(sample_input)

print("\n--- SMARTGRID AI PREDICTION ---")

print("Predicted Usage:", result["predicted_usage"])
print("Risk Status:", result["risk_status"])
print("Risk Probability:", result["risk_probability"], "%")

print("\n--- GENERATED RAG QUERY ---")
print(result["rag_query"])

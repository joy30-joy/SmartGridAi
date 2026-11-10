
from rag.smartgrid_pipeline import run_smartgrid


# Sample input
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


# Run complete pipeline
result = run_smartgrid(sample_input)


# Display results
print("\n========== FINAL RESULT ==========")

print("\nPredicted Electricity Usage:")
print(result["predicted_usage"])

print("\nRisk Status:")
print(result["risk_status"])

print("\nRisk Probability:")
print(result["risk_probability"], "%")

print("\n========== GEMINI EXPLANATION ==========")
print(result["explanation"])

print("\n========== SOURCES ==========")

for source in result["sources"]:
    print(source)

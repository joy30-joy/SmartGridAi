import json
from agentic_ai import run, to_pipeline_input, derive, ZONES
from ml.prediction import predict_grid_status

def test_agentic_extraction():
    print("=" * 65)
    print("TEST 1: Testing derive() Tool Calling Directly")
    print("=" * 65)
    derived = derive(
        temperature_mean=29.0,
        temperature_max=35.0,
        temperature_min=24.0,
        date_iso="2026-11-14",
        ercot_zone="NORTH"
    )
    print("Tool derive() output:", derived)
    assert derived["CDD"] == 10.7, f"Expected CDD 10.7, got {derived['CDD']}"
    assert derived["temp_range"] == 11.0, f"Expected temp_range 11.0, got {derived['temp_range']}"
    assert derived["weekend"] == 1, f"Expected weekend 1, got {derived['weekend']}"
    assert derived["ERCOT_zone_encoded"] == ZONES.index("NORTH")
    print("[PASS] derive() tool verification passed!")

    print("\n" + "=" * 65)
    print("TEST 2: Testing Agentic AI Extraction on User Scenario")
    print("=" * 65)
    sample_text = "Saturday Nov 14 2026, Dallas, high 35C, low 24C, avg 29C, humidity 60%, wind 4 m/s, no rain, load 52000 MW"
    print(f"Input: '{sample_text}'\n")

    result = run(sample_text)
    print("Extracted Result:")
    print(json.dumps(result, indent=2))

    required_keys = [
        "zone_usage", "temperature_mean", "temperature_max", "temperature_min",
        "humidity", "precipitation_mm", "wind_speed_mps", "snowfall_cm",
        "CDD", "temp_range", "weekend", "ERCOT_zone_encoded"
    ]

    for k in required_keys:
        assert k in result, f"Missing required key: {k}"

    assert result["temperature_mean"] == 29.0
    assert result["temperature_max"] == 35.0
    assert result["temperature_min"] == 24.0
    assert result["humidity"] == 60.0
    assert result["wind_speed_mps"] == 4.0
    assert result["precipitation_mm"] == 0.0
    assert result["zone_usage"] == 52000.0
    assert result["CDD"] == 10.7
    assert result["temp_range"] == 11.0
    assert result["weekend"] == 1
    assert result["ERCOT_zone_encoded"] == 2
    print("[PASS] All required keys and values verified successfully!")

    print("\n" + "=" * 65)
    print("TEST 3: End-to-End Prediction with ML Models")
    print("=" * 65)
    pipeline_input = to_pipeline_input(result, sample_text)
    print("Pipeline Input Formatted:")
    for k, v in pipeline_input.items():
        print(f"  {k}: {v}")

    prediction = predict_grid_status(pipeline_input)
    print("\nML Prediction Output:")
    print(f"  Predicted Usage: {prediction['predicted_usage']:,.2f} MW")
    print(f"  Usage Threshold: {prediction['usage_threshold']:,.2f} MW")
    print(f"  Risk Status: {prediction['risk_status']}")
    print(f"  High Usage Probability: {prediction['risk_probability']:.2f}%")

    assert "predicted_usage" in prediction
    assert "risk_status" in prediction
    assert "risk_probability" in prediction
    print("\n[PASS] End-to-End Agentic Prediction passed with flying colors!")

if __name__ == "__main__":
    test_agentic_extraction()

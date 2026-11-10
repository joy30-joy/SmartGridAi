import joblib

# Load regression model
regression_model = joblib.load(
    "artifacts/usage_regression_model.pkl"
)

print("Regression model loaded successfully!")
print("Type:", type(regression_model))

# Load classification model
classification_data = joblib.load(
    "artifacts/usage_classification_model.pkl"
)

print("\nClassification model loaded successfully!")
print("Model type:", type(classification_data["model"]))
print("Usage threshold:", classification_data["threshold"])

print("\nBoth models are ready for integration!")

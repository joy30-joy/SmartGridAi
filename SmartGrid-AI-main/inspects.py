import joblib
import os

model_files = [
    "artifacts/duration_model.pkl",
    "artifacts/customer_severity_model.pkl",
    "artifacts/ercot_zone_encoder.pkl"
]

for path in model_files:

    print("\n" + "=" * 50)
    print("FILE:", path)

    if not os.path.exists(path):
        print("File not found!")
        continue

    try:
        obj = joblib.load(path)

        print("STATUS: Loaded successfully")
        print("TYPE:", type(obj))

        if hasattr(obj, "feature_names_in_"):
            print("FEATURES:", list(obj.feature_names_in_))

        if hasattr(obj, "n_features_in_"):
            print("NUMBER OF FEATURES:", obj.n_features_in_)

        if hasattr(obj, "classes_"):
            print("CLASSES:", obj.classes_)

        if hasattr(obj, "categories_"):
            print("CATEGORIES:", obj.categories_)

    except Exception as e:
        print("STATUS: Loading failed")
        print("ERROR:", type(e).__name__, e)
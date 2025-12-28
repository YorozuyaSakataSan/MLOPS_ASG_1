import joblib
import pandas as pd
from pathlib import Path

def test_model_file_and_predict_proba():
    model_path = Path("models/best_model.joblib")
    assert model_path.exists(), "models/best_model.joblib not found. Run training first."

    model = joblib.load(model_path)

    sample = pd.DataFrame([{
        "age": 55,
        "sex": 1,
        "cp": 2,
        "trestbps": 140,
        "chol": 240,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.2,
        "slope": 1,
        "ca": 0,
        "thal": 2,
    }])

    proba = model.predict_proba(sample)[0][1]
    assert 0.0 <= float(proba) <= 1.0

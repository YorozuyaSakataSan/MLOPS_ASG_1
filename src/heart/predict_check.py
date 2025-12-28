import joblib
import pandas as pd

def main():
    model = joblib.load("models/best_model.joblib")

    # One sample input row (use a realistic example)
    sample = {
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
    }

    X = pd.DataFrame([sample])

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]

    print("Prediction (0=no disease, 1=disease):", int(pred))
    print("Confidence (probability of disease):", float(proba))

if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
)
import matplotlib.pyplot as plt

from src.heart.features import FeatureConfig, split_X_y, build_preprocess_pipeline


ARTIFACT_DIR = Path("artifacts")


def evaluate_model_cv(model: Pipeline, X, y) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=["accuracy", "precision", "recall", "roc_auc"],
        return_train_score=False,
    )

    # Average CV scores
    return {
        "cv_accuracy_mean": float(scores["test_accuracy"].mean()),
        "cv_precision_mean": float(scores["test_precision"].mean()),
        "cv_recall_mean": float(scores["test_recall"].mean()),
        "cv_roc_auc_mean": float(scores["test_roc_auc"].mean()),
    }


def evaluate_on_test(model: Pipeline, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred)),
        "test_recall": float(recall_score(y_test, y_pred)),
        "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }


def save_plots(model: Pipeline, X_test, y_test) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # ROC Curve
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "roc_curve.png", dpi=200)
    plt.close()


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv("data/raw/heart.csv")
    cfg = FeatureConfig()

    mlflow.set_experiment("HeartDiseaseRisk")

    X, y = split_X_y(df, cfg)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocess = build_preprocess_pipeline(cfg)

    # Model 1: Logistic Regression
    logreg = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=2000)),
        ]
    )

    # Model 2: Random Forest
    rf = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", RandomForestClassifier(n_estimators=300, random_state=42)),
        ]
    )

    results = {}
    results["config"] = asdict(cfg)

    print("\n=== Logistic Regression ===")
    with mlflow.start_run(run_name="LogisticRegression"):
        mlflow.log_param("model", "logistic_regression")
        mlflow.log_param("max_iter", 2000)

        mlflow.log_param("numeric_cols", ",".join(cfg.numeric_cols))
        mlflow.log_param("categorical_cols", ",".join(cfg.categorical_cols))

        lr_cv = evaluate_model_cv(logreg, X_train, y_train)
        logreg.fit(X_train, y_train)
        lr_test = evaluate_on_test(logreg, X_test, y_test)

        # log CV metrics
        for k, v in lr_cv.items():
            mlflow.log_metric(k, v)

        # log test metrics
        mlflow.log_metric("test_accuracy", lr_test["test_accuracy"])
        mlflow.log_metric("test_precision", lr_test["test_precision"])
        mlflow.log_metric("test_recall", lr_test["test_recall"])
        mlflow.log_metric("test_roc_auc", lr_test["test_roc_auc"])

        results["logistic_regression"] = {**lr_cv, **lr_test}

        # log model (entire pipeline)
        mlflow.sklearn.log_model(logreg, name="model")

    print("\n=== Random Forest ===")
    with mlflow.start_run(run_name="RandomForest"):
        mlflow.log_param("model", "random_forest")
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("random_state", 42)

        mlflow.log_param("numeric_cols", ",".join(cfg.numeric_cols))
        mlflow.log_param("categorical_cols", ",".join(cfg.categorical_cols))

        rf_cv = evaluate_model_cv(rf, X_train, y_train)
        rf.fit(X_train, y_train)
        rf_test = evaluate_on_test(rf, X_test, y_test)

        for k, v in rf_cv.items():
            mlflow.log_metric(k, v)

        mlflow.log_metric("test_accuracy", rf_test["test_accuracy"])
        mlflow.log_metric("test_precision", rf_test["test_precision"])
        mlflow.log_metric("test_recall", rf_test["test_recall"])
        mlflow.log_metric("test_roc_auc", rf_test["test_roc_auc"])

        results["random_forest"] = {**rf_cv, **rf_test}

        mlflow.sklearn.log_model(logreg, name="model")

    # Pick best model by test ROC-AUC
    best_name = max(
        ["logistic_regression", "random_forest"],
        key=lambda name: results[name]["test_roc_auc"],
    )
    results["best_model"] = best_name
    print("\n✅ Best model by test ROC-AUC:", best_name)

    # Save metrics
    metrics_path = ARTIFACT_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Saved metrics to:", metrics_path.resolve())

    # Save ROC curve of best model
    best_model = logreg if best_name == "logistic_regression" else rf
    save_plots(best_model, X_test, y_test)
    print("Saved ROC plot to:", (ARTIFACT_DIR / "roc_curve.png").resolve())

    with mlflow.start_run(run_name="Artifacts"):
        mlflow.log_param("best_model", best_name)
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(ARTIFACT_DIR / "roc_curve.png"))

    # Retrain best model on full dataset (recommended for final packaging)
    best_model.fit(X, y)

    # Save to models/ for reuse in API
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "best_model.joblib"
    joblib.dump(best_model, model_path)

    # Save small metadata too
    meta_path = models_dir / "model_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_model": best_name,
                "dataset_rows": int(df.shape[0]),
                "dataset_cols": int(df.shape[1]),
            },
            f,
            indent=2,
        )

    print("✅ Saved packaged model to:", model_path.resolve())
    print("✅ Saved model metadata to:", meta_path.resolve())



if __name__ == "__main__":
    main()

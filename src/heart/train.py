from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

import pandas as pd

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

    # ---- Logistic Regression ----
    print("\n=== Logistic Regression ===")
    lr_cv = evaluate_model_cv(logreg, X_train, y_train)
    logreg.fit(X_train, y_train)
    lr_test = evaluate_on_test(logreg, X_test, y_test)
    results["logistic_regression"] = {**lr_cv, **lr_test}
    print("CV:", lr_cv)
    print("TEST:", {k: lr_test[k] for k in ["test_accuracy", "test_precision", "test_recall", "test_roc_auc"]})

    # ---- Random Forest ----
    print("\n=== Random Forest ===")
    rf_cv = evaluate_model_cv(rf, X_train, y_train)
    rf.fit(X_train, y_train)
    rf_test = evaluate_on_test(rf, X_test, y_test)
    results["random_forest"] = {**rf_cv, **rf_test}
    print("CV:", rf_cv)
    print("TEST:", {k: rf_test[k] for k in ["test_accuracy", "test_precision", "test_recall", "test_roc_auc"]})

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


if __name__ == "__main__":
    main()

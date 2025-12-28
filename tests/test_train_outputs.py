import json
from pathlib import Path

def test_metrics_artifact_exists_and_has_keys():
    metrics_path = Path("artifacts/metrics.json")
    assert metrics_path.exists(), "artifacts/metrics.json not found. Run training first."

    data = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert "logistic_regression" in data
    assert "random_forest" in data
    assert "best_model" in data

    # check a few important metrics exist
    lr = data["logistic_regression"]
    assert "test_roc_auc" in lr
    assert "cv_accuracy_mean" in lr

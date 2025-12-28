import pandas as pd
from src.heart.features import FeatureConfig, split_X_y, build_preprocess_pipeline

def test_preprocess_fit_transform():
    df = pd.read_csv("data/raw/heart.csv")
    cfg = FeatureConfig()

    X, y = split_X_y(df, cfg)
    preprocess = build_preprocess_pipeline(cfg)

    Xt = preprocess.fit_transform(X)

    # Basic sanity checks
    assert X.shape[0] == df.shape[0]
    assert Xt.shape[0] == df.shape[0]
    assert set(y.unique()).issubset({0, 1})

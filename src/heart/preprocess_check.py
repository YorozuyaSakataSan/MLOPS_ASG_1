import pandas as pd
from src.heart.features import FeatureConfig, split_X_y, build_preprocess_pipeline

def main():
    df = pd.read_csv("data/raw/heart.csv")
    cfg = FeatureConfig()

    X, y = split_X_y(df, cfg)
    preprocess = build_preprocess_pipeline(cfg)

    Xt = preprocess.fit_transform(X)

    print("✅ Preprocess fit successful")
    print("X shape:", X.shape)
    print("Transformed shape:", Xt.shape)
    print("y distribution:\n", y.value_counts())

if __name__ == "__main__":
    main()

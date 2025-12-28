from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class FeatureConfig:
    target_col: str = "target"

    # Based on your downloaded dataset columns
    numeric_cols: Tuple[str, ...] = (
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak",
    )

    categorical_cols: Tuple[str, ...] = (
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal",
    )


def split_X_y(df: pd.DataFrame, cfg: FeatureConfig) -> tuple[pd.DataFrame, pd.Series]:
    if cfg.target_col not in df.columns:
        raise ValueError(f"Target column '{cfg.target_col}' not found in columns: {df.columns.tolist()}")
    X = df.drop(columns=[cfg.target_col]).copy()
    y = df[cfg.target_col].copy()
    return X, y


def build_preprocess_pipeline(cfg: FeatureConfig) -> ColumnTransformer:
    """
    Returns a ColumnTransformer:
      - numeric: impute median + standard scale
      - categorical: impute most frequent + one hot encode
    """

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, list(cfg.numeric_cols)),
            ("cat", categorical_pipe, list(cfg.categorical_cols)),
        ],
        remainder="drop",
    )

    return preprocess

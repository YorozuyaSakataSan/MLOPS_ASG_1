from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo


def main() -> None:
    """
    Downloads the Heart Disease dataset from UCI via ucimlrepo,
    creates a single DataFrame, converts the target to binary,
    and saves it to data/raw/heart.csv
    """

    # 1) Fetch dataset (UCI Heart Disease)
    # Dataset id for Heart Disease is 45 in ucimlrepo
    dataset = fetch_ucirepo(id=45)

    X = dataset.data.features
    y = dataset.data.targets

    # 2) Combine features + target into one DataFrame
    df = pd.concat([X, y], axis=1)

    # 3) Standardize column names (optional but helpful)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # 4) Convert target to binary
    # In UCI heart disease, target (often called "num") is 0-4.
    # 0 = absence, 1-4 = presence
    # Some versions name it "num", others "target". We'll handle both.
    target_col = None
    for c in df.columns:
        if c in ["num", "target"]:
            target_col = c
            break

    if target_col is None:
        raise ValueError(f"Could not find target column. Columns found: {df.columns.tolist()}")

    df["target"] = df[target_col].apply(lambda v: 0 if int(v) == 0 else 1)

    # Drop the old target column if it wasn't already called "target"
    if target_col != "target":
        df = df.drop(columns=[target_col])

    # 5) Save
    out_path = Path("data/raw/heart.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("✅ Download complete!")
    print(f"Saved to: {out_path.resolve()}")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Target value counts:\n", df["target"].value_counts())


if __name__ == "__main__":
    main()

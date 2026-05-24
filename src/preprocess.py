import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def preprocess_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    required_columns = [
        "age", "bp", "sg", "al", "bgr",
        "sc", "sod", "pot", "bu", "hemo", "stage"
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[required_columns].copy()

    df.replace(["?", "", " "], np.nan, inplace=True)

    stage_map = {
        "s1": 0,
        "s2": 1,
        "s3": 2,
        "s4": 3,
        "s5": 4
    }

    df["stage"] = df["stage"].astype(str).str.lower().str.strip()
    df = df[df["stage"].isin(stage_map.keys())]
    df["stage"] = df["stage"].map(stage_map)

    feature_cols = [
        "age", "bp", "sg", "al", "bgr",
        "sc", "sod", "pot", "bu", "hemo"
    ]

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    X = df[feature_cols]
    y = df["stage"].astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, feature_cols, scaler
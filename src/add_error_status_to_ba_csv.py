import os
from glob import glob

import numpy as np
import pandas as pd


THRESHOLDS = (1.0, 0.5)


def _status(y_true, y_pred, threshold):
    diff = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    return np.where(diff <= float(threshold), 1, 2).astype(int)


def add_status_columns(df, thresholds=THRESHOLDS):
    if "y_val_true" not in df.columns or "y_val_pred" not in df.columns:
        raise ValueError("CSV 必须包含列：y_val_true, y_val_pred")

    y_true = df["y_val_true"]
    y_pred = df["y_val_pred"]

    df = df.copy()
    df["Prediction_Status"] = _status(y_true, y_pred, thresholds[0])
    df["Prediction_Status_0_5D"] = _status(y_true, y_pred, thresholds[1])
    return df


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ba_dir = os.path.join(project_root, "baresults")

    patterns = [
        os.path.join(ba_dir, "badata_*_full.csv"),
        os.path.join(ba_dir, "badata_*_full*.csv"),
    ]
    files = []
    for p in patterns:
        files.extend(glob(p))
    files = sorted(set(files))

    if not files:
        raise FileNotFoundError(
            "No BA result CSV found: baresults/badata_*_full.csv\n"
            "Run baanalysis.py first to generate the original CSV (y_val_true, y_val_pred)."
        )

    for in_path in files:
        df = pd.read_csv(in_path)
        out_df = add_status_columns(df, thresholds=THRESHOLDS)

        root, ext = os.path.splitext(in_path)
        out_path = f"{root}_with_status{ext}"
        out_df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"Generated: {out_path}")


if __name__ == "__main__":
    main()

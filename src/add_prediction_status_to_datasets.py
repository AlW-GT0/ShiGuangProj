import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from train_models import FULL_FEATURES, TARGET, TRAIN_DATA_PATH, VAL_DATA_PATH, load_data, preprocess_data


def _status(y_true, y_pred, threshold):
    diff = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    return np.where(diff <= float(threshold), 1, 2).astype(int)


def main(model_name="SVR", feature_set_name="full", thresholds=(1.0, 0.5)):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "models", f"{model_name}_{feature_set_name}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件：{model_path}")

    train_df, val_df = load_data()

    X_train, y_train = preprocess_data(train_df, FULL_FEATURES)
    X_val, y_val = preprocess_data(val_df, FULL_FEATURES)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = joblib.load(model_path)
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)

    out_train = train_df.copy()
    out_val = val_df.copy()

    out_train["Prediction_Status"] = _status(y_train, y_train_pred, thresholds[0])
    out_train["Prediction_Status_0_5D"] = _status(y_train, y_train_pred, thresholds[1])
    out_val["Prediction_Status"] = _status(y_val, y_val_pred, thresholds[0])
    out_val["Prediction_Status_0_5D"] = _status(y_val, y_val_pred, thresholds[1])

    results_dir = os.path.dirname(TRAIN_DATA_PATH)
    train_out_path = os.path.join(results_dir, "train_dataset_with_status.xlsx")
    val_out_path = os.path.join(results_dir, "validation_dataset_with_status.xlsx")

    out_train.to_excel(train_out_path, index=False)
    out_val.to_excel(val_out_path, index=False)

    print("已生成新文件（原文件不变）：")
    print(f"- {train_out_path}")
    print(f"- {val_out_path}")
    print(f"模型：{model_name}_{feature_set_name}.joblib")
    print(f"阈值：±{thresholds[0]:.2f}D（Prediction_Status），±{thresholds[1]:.2f}D（Prediction_Status_0_5D）")


if __name__ == "__main__":
    main()


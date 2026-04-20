# 代码修改记录

## 需求描述
需要识别模型预测效果不佳的样本，并在数据集结果表中进行标记（不覆盖原文件）。

具体要求：
1. 基于 `results/train_dataset.xlsx` 与 `results/validation_dataset.xlsx` 各生成一个新文件，并新增两列：
   - `Prediction_Status`：误差阈值 ±1.00D
   - `Prediction_Status_0_5D`：误差阈值 ±0.50D
2. 标记规则（两列一致，仅阈值不同）：
   - **1**：预测误差在阈值以内（预测较好）。
   - **2**：预测误差超过阈值（预测较差）。

## 解决方案与代码修改（便于协作合并）

### 1. 新增文件
新增 `add_prediction_status_to_datasets.py`（独立脚本，不改动 `train_models.py`），用于在两个数据集表的基础上生成“带标记的新 xlsx”。输出文件名固定为：
- `results/train_dataset_with_status.xlsx`
- `results/validation_dataset_with_status.xlsx`

### 2. 修改位置标记
为实现“±1.00D 与 ±0.50D 两列”，本次改动仅在一个新文件中完成：
1. `add_prediction_status_to_datasets.py`
   - 读取 `results/train_dataset.xlsx` 与 `results/validation_dataset.xlsx`
   - 使用 `models/SVR_full.joblib` 对两个表分别预测
   - 新增两列：`Prediction_Status` 与 `Prediction_Status_0_5D`
   - 写出两个新文件（原文件不变）

### 3. 运行方式
在 `vision_screening_baseline2` 目录运行：
- `python .\\src\\add_prediction_status_to_datasets.py`

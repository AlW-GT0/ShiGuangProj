# 代码修改记录

## 需求描述
需要识别模型预测效果不佳的样本（预测误差在 ±1.00D 之外），并在训练集和验证集数据文件中进行标记。

具体要求：
1. 在 `train_dataset.xlsx` 和 `validation_dataset.xlsx` 文件末尾添加一列 `Prediction_Status`。
2. 标记规则：
   - **1**：预测误差在 ±1.00D 以内（预测准确）。
   - **2**：预测误差超过 ±1.00D（预测偏差大）。

## 解决方案与代码修改

### 1. 修改 `train_and_evaluate` 函数
- **目的**：获取模型的具体预测值用于后续计算。
- **修改内容**：使其除了返回评估指标（MAE, RMSE等）外，还能返回具体的预测值字典：
  ```python
  predictions = {
      'train': y_train_pred,
      'val': y_val_pred
  }
  return results, predictions
  ```

### 2. 修改 `main` 主函数
- **目的**：基于最佳模型的效果对数据进行标记并保存。
- **修改内容**：
  1. **自动记录最佳模型**：在模型训练循环中，比较并记录验证集 MAE 最小的模型的预测结果。
  2. **计算误差与生成状态列**：
     - 在所有模型训练结束后，计算最佳模型在训练集和验证集上的绝对预测误差。
     - 根据误差生成新列 `Prediction_Status`：
       - `np.where(abs(diff) <= 1.0, 1, 2)`
  3. **保存文件**：
     - 将包含新列的 DataFrame 直接覆盖保存回 `train_dataset.xlsx` 和 `validation_dataset.xlsx`。

import os
import json
import joblib
import numpy as np
import pandas as pd

from config import FEATURE_MAPPING, SER_GROUPS


# 目录常量（与训练脚本一致）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODELS_DIR = os.path.join(BASE_DIR, 'models')


# 特征集（与训练脚本一致，保持顺序）
BASIC_FEATURES = [
    FEATURE_MAPPING['Age'],
    FEATURE_MAPPING['Gender'],
    FEATURE_MAPPING['Wearing refractive correction'],
    FEATURE_MAPPING['Uncorrected viosual acuity'],
    FEATURE_MAPPING['District'],
    FEATURE_MAPPING['Non-cycloplegic SER']
]

FULL_FEATURES = BASIC_FEATURES + [
    FEATURE_MAPPING['AL'],
    FEATURE_MAPPING['Kf'],
    FEATURE_MAPPING['Ks'],
    FEATURE_MAPPING['AL/CR'],
    FEATURE_MAPPING['ACD']
]

TARGET = FEATURE_MAPPING['Cycloplegic SER']


# 年龄亚组（按用户说明）
AGE_SUBGROUPS = {
    # '学龄前(4-5岁)': (4, 5),
    '小学(5-11岁)': (5, 11),
    '初中(12-15岁)': (12, 15),
    '高中(16-18岁)': (16, 18)
}


def load_data():
    train_path = os.path.join(RESULTS_DIR, 'train_dataset.xlsx')
    val_path = os.path.join(RESULTS_DIR, 'validation_dataset.xlsx')
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError('train_dataset.xlsx 或 validation_dataset.xlsx 未找到，请先生成数据集。')
    train_df = pd.read_excel(train_path)
    val_df = pd.read_excel(val_path)
    return train_df, val_df


def preprocess_data(df, features):
    X = df[features].copy()
    y = df[TARGET].copy()
    # 性别映射（与训练脚本一致）
    if FEATURE_MAPPING['Gender'] in features:
        X[FEATURE_MAPPING['Gender']] = X[FEATURE_MAPPING['Gender']].map({'男': 0, '女': 1})
    # 裸眼视力转为数值（与映射名一致）
    if FEATURE_MAPPING['Uncorrected viosual acuity'] in features:
        X[FEATURE_MAPPING['Uncorrected viosual acuity']] = pd.to_numeric(
            X[FEATURE_MAPPING['Uncorrected viosual acuity']], errors='coerce'
        )
    # 缺失值处理（数值列均值填充）
    X = X.fillna(X.mean())
    return X, y


def compute_error_metrics(y_true, y_pred):
    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    n = len(y_true)
    me = float(np.mean(errors)) if n > 0 else np.nan
    mae = float(np.mean(abs_errors)) if n > 0 else np.nan
    rmse = float(np.sqrt(np.mean(errors ** 2))) if n > 0 else np.nan
    within_0_5 = float(np.mean(abs_errors <= 0.5)) if n > 0 else np.nan
    within_1_0 = float(np.mean(abs_errors <= 1.0)) if n > 0 else np.nan
    return {
        'N': int(n),
        'ME': me,
        'MAE': mae,
        'RMSE': rmse,
        'P(|err|<=0.50D)': within_0_5,
        'P(|err|<=1.00D)': within_1_0
    }


def label_age_group(age_val):
    if pd.isna(age_val):
        return None
    try:
        age = float(age_val)
    except Exception:
        return None
    for label, (lo, hi) in AGE_SUBGROUPS.items():
        if age >= lo and age <= hi:
            return label
    if age > 15:
        return '高中(>15岁)'
    return None


def label_ser_group(ser_val):
    if pd.isna(ser_val):
        return None
    try:
        v = float(ser_val)
    except Exception:
        return None
    for label, (lo, hi) in SER_GROUPS.items():
        if v > lo and v <= hi:
            return label
    return None



def analyze_subgroups(df, features, target_col, model, scaler, group_type):
    data = df.copy()
    if group_type == 'age':
        age_col = FEATURE_MAPPING['Age']
        data['__group__'] = data[age_col].apply(label_age_group)
        group_labels = list(AGE_SUBGROUPS.keys())
    elif group_type == 'ser':
        data['__group__'] = data[target_col].apply(label_ser_group)
        group_labels = list(SER_GROUPS.keys())
    else:
        raise ValueError('Unknown group_type')

    results = []
    for g in group_labels:
        subset = data[data['__group__'] == g]
        if subset.empty:
            results.append({'Group': g, 'N': 0, 'ME': np.nan, 'MAE': np.nan, 'RMSE': np.nan,
                            'P(|err|<=0.50D)': np.nan, 'P(|err|<=1.00D)': np.nan})
            continue
        X, y = preprocess_data(subset, features)
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        metrics = compute_error_metrics(y.values, y_pred)
        results.append({'Group': g, **metrics})
    return results


def load_best_model_by_results():
    result_json = os.path.join(RESULTS_DIR, 'model_results.json')
    if not os.path.exists(result_json):
        raise FileNotFoundError('model_results.json 未找到，请先运行训练脚本生成模型评估结果。')
    with open(result_json, 'r', encoding='utf-8') as f:
        results = json.load(f)
    # print(results)
    # 选择验证集 MAE 最低的模型
    best = sorted(results, key=lambda x: x['val_mae'])[0]
    model_path = os.path.join(MODELS_DIR, best['model_file'])
    scaler_path = os.path.join(MODELS_DIR, best['scaler_file'])
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_set = best['feature_set']
    return best, model, scaler, feature_set


def main():
    print('开始亚组分析...')
    train_df, val_df = load_data()
    best_info, model, scaler, feature_set = load_best_model_by_results()
    print(f"使用最佳模型: {best_info['model_name']} ({feature_set})")

    features_for_best = BASIC_FEATURES if feature_set == 'basic' else FULL_FEATURES

    # 训练集与验证集的年龄/SER分组分析
    age_train = analyze_subgroups(train_df, features_for_best, TARGET, model, scaler, 'age')
    age_val = analyze_subgroups(val_df, features_for_best, TARGET, model, scaler, 'age')

    ser_train = analyze_subgroups(train_df, features_for_best, TARGET, model, scaler, 'ser')
    ser_val = analyze_subgroups(val_df, features_for_best, TARGET, model, scaler, 'ser')

    # 保存Excel
    out_xlsx = os.path.join(RESULTS_DIR, 'subgroup_analysis.xlsx')
    with pd.ExcelWriter(out_xlsx) as writer:
        pd.DataFrame(age_train).to_excel(writer, sheet_name='Age_Train', index=False)
        pd.DataFrame(age_val).to_excel(writer, sheet_name='Age_Validation', index=False)
        pd.DataFrame(ser_train).to_excel(writer, sheet_name='SER_Train', index=False)
        pd.DataFrame(ser_val).to_excel(writer, sheet_name='SER_Validation', index=False)

    # 文本报告
    out_txt = os.path.join(RESULTS_DIR, 'subgroup_report.txt')
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write('散瞳后屈光度预测模型亚组分析\n')
        f.write('=' * 50 + '\n\n')
        f.write(f"最佳模型: {best_info['model_name']} (特征集: {feature_set})\n\n")

        def write_block(title, rows):
            f.write(title + '\n')
            f.write('-' * 30 + '\n')
            for r in rows:
                f.write(
                    f"{r['Group']}: N={r['N']}, ME={r['ME']:.4f}, MAE={r['MAE']:.4f}, "
                    f"RMSE={r['RMSE']:.4f}, P(|err|<=0.50D)={r['P(|err|<=0.50D)']:.2%}, "
                    f"P(|err|<=1.00D)={r['P(|err|<=1.00D)']:.2%}\n"
                )
            f.write('\n')

        write_block('年龄(训练集)', age_train)
        write_block('年龄(验证集)', age_val)
        write_block('SER(训练集)', ser_train)
        write_block('SER(验证集)', ser_val)

    print(f"完成！输出: {out_xlsx} 与 {out_txt}")


if __name__ == '__main__':
    main()
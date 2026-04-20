import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LassoCV, LinearRegression
import xgboost as xgb
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 导入配置
from config import FEATURE_MAPPING

plt.rcParams['font.sans-serif'] = ['SimHei']  #
plt.rcParams['axes.unicode_minus'] = False  #

# 定义常量
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
TRAIN_DATA_PATH = os.path.join(RESULTS_DIR, 'train_dataset.xlsx')
VAL_DATA_PATH = os.path.join(RESULTS_DIR, 'validation_dataset.xlsx')

# 确保模型目录存在
os.makedirs(MODELS_DIR, exist_ok=True)

# 定义两组特征
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

# 目标变量
TARGET = FEATURE_MAPPING['Cycloplegic SER']

# 定义模型
def get_models():
    """获取所有模型"""
    models = {
        'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1),
        'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
        'LassoCV': LassoCV(cv=5, random_state=42),
        'Linear': LinearRegression()
    }
    return models

def load_data():
    """加载数据"""
    print("加载数据...")
    train_df = pd.read_excel(TRAIN_DATA_PATH)
    val_df = pd.read_excel(VAL_DATA_PATH)
    return train_df, val_df

def preprocess_data(df, features):
    """预处理数据"""
    # 选择特征和目标变量
    X = df[features].copy()
    y = df[TARGET].copy()
    
    # 处理分类特征
    if FEATURE_MAPPING['Gender'] in features:
        X[FEATURE_MAPPING['Gender']] = X[FEATURE_MAPPING['Gender']].map({'男': 0, '女': 1})
    
    
    # 处理视力值 (Uncorrected visual acuity)
    if 'Uncorrected visual acuity' in features:
        # 将视力值转换为数值
        X['Uncorrected visual acuity'] = pd.to_numeric(X['Uncorrected visual acuity'], errors='coerce')
    
    # 处理缺失值
    X = X.fillna(X.mean())
    
    return X, y

def train_and_evaluate(X_train, y_train, X_val, y_val, model_name, model, feature_set_name):
    """训练和评估模型"""
    print(f"训练 {model_name} 模型 (特征集: {feature_set_name})...")
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 训练模型
    model.fit(X_train_scaled, y_train)
    
    # 预测
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)
    
    # 计算评估指标
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_r2 = r2_score(y_val, y_val_pred)
    
    # 保存模型
    model_filename = f"{model_name}_{feature_set_name}.joblib"
    scaler_filename = f"{model_name}_{feature_set_name}_scaler.joblib"
    joblib.dump(model, os.path.join(MODELS_DIR, model_filename))
    joblib.dump(scaler, os.path.join(MODELS_DIR, scaler_filename))
    
    # 计算特征重要性（如果模型支持）
    feature_importance = {}
    if hasattr(model, 'feature_importances_'):
        # 根据config中的映射，将原始列名转为映射后的列名
        reverse_mapping = {v: k for k, v in FEATURE_MAPPING.items()}
        feature_names = [reverse_mapping.get(col, col) for col in X_train.columns]
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        for i in range(len(feature_names)):
            feature_importance[feature_names[indices[i]]] = float(importances[indices[i]])
    elif model_name == 'Linear' or model_name == 'LassoCV':
        # 根据config中的映射，将原始列名转为映射后的列名
        reverse_mapping = {v: k for k, v in FEATURE_MAPPING.items()}
        feature_names = [reverse_mapping.get(col, col) for col in X_train.columns]  
        importances = np.abs(model.coef_)
        indices = np.argsort(importances)[::-1]
        
        for i in range(len(feature_names)):
            feature_importance[feature_names[indices[i]]] = float(importances[indices[i]])
    
    # 返回结果
    results = {
        'model_name': model_name,
        'feature_set': feature_set_name,
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'val_mae': val_mae,
        'val_rmse': val_rmse,
        'val_r2': val_r2,
        'feature_importance': feature_importance,
        'model_file': model_filename,
        'scaler_file': scaler_filename
    }
    
    return results

def plot_feature_importance(results, output_dir):
    """绘制特征重要性图"""
    for result in results:
        if not result['feature_importance']:
            continue
            
        model_name = result['model_name']
        feature_set = result['feature_set']
        
        # 排序特征重要性
        importances = result['feature_importance']
        sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)        
        features = [x[0] for x in sorted_importances]
        values = [x[1] for x in sorted_importances]
        
        # 绘图
        plt.figure(figsize=(10, 6))
        plt.barh(features, values)
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.title(f'{model_name} - {feature_set} Feature Importance')
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(os.path.join(output_dir, f'{model_name}_{feature_set}_feature_importance.png'))
        plt.close()


def generate_report(results, output_dir):
    """生成报告"""
    # 保存结果为JSON
    with open(os.path.join(output_dir, 'model_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    # 创建结果表格
    results_df = []
    for result in results:
        row = {
            '模型': result['model_name'],
            '特征集': result['feature_set'],
            '训练集MAE': round(result['train_mae'], 4),
            '训练集RMSE': round(result['train_rmse'], 4),
            '训练集R²': round(result['train_r2'], 4),
            '验证集MAE': round(result['val_mae'], 4),
            '验证集RMSE': round(result['val_rmse'], 4),
            '验证集R²': round(result['val_r2'], 4)
        }
        results_df.append(row)
    
    results_df = pd.DataFrame(results_df)
    results_df.to_excel(os.path.join(output_dir, 'model_comparison.xlsx'), index=False)
    
    # 生成文本报告
    with open(os.path.join(output_dir, 'model_report.txt'), 'w', encoding='utf-8') as f:
        f.write("散瞳后屈光度预测模型评估报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 基本信息
        f.write("1. 基本信息\n")
        f.write("-" * 30 + "\n")
        f.write(f"目标变量: {TARGET}\n")
        f.write("基本特征集: " + ", ".join(BASIC_FEATURES) + "\n")
        f.write("完整特征集: " + ", ".join(FULL_FEATURES) + "\n\n")
        
        # 模型性能比较
        f.write("2. 模型性能比较\n")
        f.write("-" * 30 + "\n")
        
        # 按验证集MAE排序
        sorted_results = sorted(results, key=lambda x: x['val_mae'])
        
        for result in sorted_results:
            model_name = result['model_name']
            feature_set = result['feature_set']
            
            f.write(f"{model_name} ({feature_set}):\n")
            f.write(f"  训练集: MAE={result['train_mae']:.4f}, RMSE={result['train_rmse']:.4f}, R²={result['train_r2']:.4f}\n")
            f.write(f"  验证集: MAE={result['val_mae']:.4f}, RMSE={result['val_rmse']:.4f}, R²={result['val_r2']:.4f}\n")
            
            # 特征重要性
            if result['feature_importance']:
                f.write("  特征重要性:\n")
                sorted_importance = sorted(result['feature_importance'].items(), key=lambda x: x[1], reverse=True)
                for feature, importance in sorted_importance:
                    f.write(f"    {feature}: {importance:.4f}\n")
            
            f.write("\n")
        
        # 结论
        f.write("3. 结论\n")
        f.write("-" * 30 + "\n")
        best_model = sorted_results[0]
        f.write(f"最佳模型: {best_model['model_name']} (特征集: {best_model['feature_set']})\n")
        f.write(f"验证集性能: MAE={best_model['val_mae']:.4f}, RMSE={best_model['val_rmse']:.4f}, R²={best_model['val_r2']:.4f}\n\n")
        
        # 比较两组特征集的效果
        basic_results = [r for r in results if r['feature_set'] == 'basic']
        full_results = [r for r in results if r['feature_set'] == 'full']
        
        if basic_results and full_results:
            best_basic = sorted(basic_results, key=lambda x: x['val_mae'])[0]
            best_full = sorted(full_results, key=lambda x: x['val_mae'])[0]
            
            f.write("特征集比较:\n")
            f.write(f"  基本特征集最佳模型: {best_basic['model_name']}, MAE={best_basic['val_mae']:.4f}\n")
            f.write(f"  完整特征集最佳模型: {best_full['model_name']}, MAE={best_full['val_mae']:.4f}\n")
            
            if best_full['val_mae'] < best_basic['val_mae']:
                improvement = (best_basic['val_mae'] - best_full['val_mae']) / best_basic['val_mae'] * 100
                f.write(f"  添加眼轴、角膜曲率、轴率比和前房深度特征使预测MAE降低了 {improvement:.2f}%\n")
            else:
                f.write("  添加额外特征未能提高预测性能\n")

def main():
    """主函数"""
    print("开始散瞳后屈光度预测模型训练与评估...")
    
    # 加载数据
    train_df, val_df = load_data()
    print(f'{train_df.shape},{val_df.shape}')
    # 获取模型
    models = get_models()
    
    # 存储结果
    all_results = []
    
    # 基本特征集
    print("\n使用基本特征集...")
    X_train_basic, y_train = preprocess_data(train_df, BASIC_FEATURES)
    X_val_basic, y_val = preprocess_data(val_df, BASIC_FEATURES)
    
    for model_name, model in models.items():
        result = train_and_evaluate(
            X_train_basic, y_train, X_val_basic, y_val, 
            model_name, model, 'basic'
        )
        all_results.append(result)
        print(f"  {model_name}: 验证集 MAE={result['val_mae']:.4f}, RMSE={result['val_rmse']:.4f}, R²={result['val_r2']:.4f}")
    
    # 完整特征集
    print("\n使用完整特征集...")
    X_train_full, y_train = preprocess_data(train_df, FULL_FEATURES)
    X_val_full, y_val = preprocess_data(val_df, FULL_FEATURES)
    
    for model_name, model in models.items():
        result = train_and_evaluate(
            X_train_full, y_train, X_val_full, y_val, 
            model_name, model, 'full'
        )
        all_results.append(result)
        print(f"  {model_name}: 验证集 MAE={result['val_mae']:.4f}, RMSE={result['val_rmse']:.4f}, R²={result['val_r2']:.4f}")

    # 生成可视化和报告
    print("\n生成结果报告和可视化...")
    plot_feature_importance(all_results, RESULTS_DIR)
    generate_report(all_results, RESULTS_DIR)
    
    print(f"\n完成！结果已保存到 {RESULTS_DIR} 目录")

if __name__ == "__main__":
    main()

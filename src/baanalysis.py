from train_models import *

def evaluate(X_train, y_train, X_val, y_val, model_name, model, feature_set_name):
    """评估模型"""
    print(f"评估 {model_name} 模型 (特征集: {feature_set_name})...")
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
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
    
    # 生成文件名
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(project_root, "baresults")
    os.makedirs(save_dir, exist_ok=True)
    filename = f"badata_{model_name}_{feature_set_name}.csv"
    filepath = os.path.join(save_dir, filename)
    
    # 创建DataFrame并保存
    pred_df = pd.DataFrame({
        'y_val_true': y_val,
        'y_val_pred': y_val_pred
    })
    # 保存到CSV
    pred_df.to_csv(filepath, index=False, encoding='utf-8')
    print(f"预测结果已保存到: {filepath}")

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
        # 'file': model_filename,
    }
    return results
    

def baa():
    # 加载数据
    train_df, val_df = load_data()
    print(f'{train_df.shape},{val_df.shape}')

    model = joblib.load('./models/SVR_full.joblib')

    model_name = 'SVR'

    # 存储结果
    all_results = []
    
    print("\n使用完整特征集...")
    X_train_full, y_train = preprocess_data(train_df, FULL_FEATURES)
    X_val_full, y_val = preprocess_data(val_df, FULL_FEATURES)
    
    result = evaluate(
        X_train_full, y_train, X_val_full, y_val, 
        model_name, model, 'full'
    )
    print(f"  {model_name}: 验证集 MAE={result['val_mae']:.4f}, RMSE={result['val_rmse']:.4f}, R2={result['val_r2']:.4f}")


if __name__ == "__main__":
    baa()

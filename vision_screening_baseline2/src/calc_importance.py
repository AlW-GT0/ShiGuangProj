import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

from config import FULL_FEATURES, TARGET, MODEL_LIST
from train_models import preprocess_data

# 设置中文字体支持
from font_setup_cn import setup_matplotlib_cn_font
font_prop, font_path = setup_matplotlib_cn_font("./data/fonts")
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

# 定义模型路径
result_dir = './results/shap/'
model_dir = './models/'
os.makedirs(result_dir, exist_ok=True)

for modelname in MODEL_LIST:
    model_path = os.path.join(model_dir, f'{modelname}_full.joblib')
    scaler_path = os.path.join(model_dir, f'{modelname}_full_scaler.joblib')

    # 1. 加载模型和标准化器
    print("正在加载模型和标准化器...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print(f"模型类型: {type(model)}")
    print(f"模型加载完成")

    # 2. 加载数据（请根据您的实际数据路径修改）
    # 这里假设您的数据文件在某个位置，需要您根据实际情况调整
    print("\n正在加载数据...")
    # 请根据您的实际数据文件路径和格式修改这部分
    data_path = './data/doubledata.xlsx'  # 示例路径
    data = pd.read_excel(data_path)
    print("\n使用完整特征集...")
    X_full, y = preprocess_data(data, FULL_FEATURES)

    # 如不知道具体数据，这里提供一个示例数据加载方式
    # 如有训练数据，请取消上面的注释并修改为正确的路径
    # print("请根据实际情况修改数据加载部分！")
    # 临时：如果您需要测试，可以使用以下代码生成示例数据
    # 注意：这只是一个示例，实际分析应该使用您的真实数据
    # np.random.seed(42)
    # n_samples = 1000
    # n_features = 20
    # X_dummy = np.random.randn(n_samples, n_features)
    # feature_names = [f'feature_{i}' for i in range(n_features)]
    # X = pd.DataFrame(X_dummy, columns=feature_names)

    # 3. 数据标准化（使用加载的scaler）
    print("\n正在对数据进行标准化...")
    # 为了节省时间，可以只选取部分样本
    X_sample = X_full[:100]
    print(type(X_sample))
    SampleDataPath = f'{result_dir}sampledata.csv'
    X_sample.to_csv(SampleDataPath, encoding='utf-8')
    X_scaled = scaler.transform(X_sample)

    # 4. 创建SHAP解释器
    print("\n创建SHAP解释器...")
    # 对于线性模型，可以使用LinearExplainer
    # explainer = shap.LinearExplainer(model, X_scaled, feature_perturbation="correlation_dependent")
    # 使用KernelExplainer（更通用但较慢）
    explainer = shap.KernelExplainer(model.predict, X_scaled)
    # 5. 计算SHAP值
    print("计算SHAP值（这可能需要一些时间）...")
    shap_values = explainer.shap_values(X_scaled)

    # 6. 可视化SHAP结果
    print("\n生成SHAP可视化图表...")

    # # 6.1 汇总图（蜂群图）
    # plt.figure(figsize=(12, 8))
    # shap.summary_plot(shap_values, X_sample, feature_names=FULL_FEATURES, show=False)
    # plt.title('SHAP Feature Importance Summary (LassoCV Full Model)')
    # plt.tight_layout()
    # plt.savefig(f'{result_dir}/shap_summary_full.png', dpi=300, bbox_inches='tight')
    # plt.show()

    # 计算特征重要性并获取前十的特征
    mean_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        'feature': FULL_FEATURES,
        'importance': mean_shap
    }).sort_values('importance', ascending=False)
    # 获取前十的特征
    top10_features = feature_importance.head(10)['feature'].values
    top10_importance = feature_importance.head(10)['importance'].values

    # 6.2 条形图（平均绝对SHAP值）
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values[:, feature_importance.head(10).index], 
                  X_sample.iloc[:, feature_importance.head(10).index], 
                  feature_names=top10_features, 
                  plot_type="bar", show=False)
    plt.title(f'SHAP Feature Importance Bar Plot ({modelname} Full Model)')
    plt.tight_layout()
    savepath = f'{result_dir}/shap_bar_{modelname}_full.png'
    plt.savefig(savepath, dpi=300, bbox_inches='tight')
    # plt.show()
    print(f"\n特征重要性结果已保存到 \'{savepath}\'")

    # 6.3 输出特征重要性排序
    print("\n特征重要性排序（基于平均绝对SHAP值）:")
    print(feature_importance.to_string(index=False))

    # # 7. 保存特征重要性结果到CSV
    # savepath = f'{result_dir}/shap_feature_importance_{modelname}_full.csv'
    # feature_importance.to_csv(savepath, index=False)
    # print(f"\n特征重要性结果已保存到 \'{savepath}\'")

    # # 8. 可选：创建单个特征的依赖图
    # print("\n为Top 5重要特征生成依赖图...")
    # top_features = feature_importance.head(5)['feature'].tolist()
    # for feature in top_features:
    #     feature_idx = FULL_FEATURES.index(feature)
    #     plt.figure(figsize=(10, 6))
    #     shap.dependence_plot(feature_idx, shap_values, X_sample, 
    #                          feature_names=FULL_FEATURES, show=False)
    #     plt.title(f'SHAP Dependence Plot: {feature}')
    #     plt.tight_layout()
    #     plt.savefig(f'{result_dir}/shap_dependence_{feature}.png', dpi=300, bbox_inches='tight')
    #     plt.show()

print("\nSHAP分析完成！")

import os
import pandas as pd
import numpy as np

from config import FEATURE_MAPPINGD, FEATURE_MAPPINGS
DFEATURES = [
    FEATURE_MAPPINGD['Age'],
    FEATURE_MAPPINGD['Gender'],
    FEATURE_MAPPINGD['Wearing refractive correction'],
    FEATURE_MAPPINGD['Uncorrected viosual acuity'],
    FEATURE_MAPPINGD['Non-cycloplegic SER'],
    FEATURE_MAPPINGD['Cycloplegic SER'],
    FEATURE_MAPPINGD['District'],
    FEATURE_MAPPINGD['AL'],
    FEATURE_MAPPINGD['Kf'],
    FEATURE_MAPPINGD['Ks'],
    FEATURE_MAPPINGD['AL/CR'],
    FEATURE_MAPPINGD['ACD'],
]
SFEATURES = [
    FEATURE_MAPPINGS['Age'],
    FEATURE_MAPPINGS['Gender'],
    FEATURE_MAPPINGS['Wearing refractive correction'],
    FEATURE_MAPPINGS['Uncorrected viosual acuity'],
    FEATURE_MAPPINGS['Non-cycloplegic SER'],
    FEATURE_MAPPINGS['Cycloplegic SER'],
    FEATURE_MAPPINGS['District'],
    FEATURE_MAPPINGS['AL'],
    FEATURE_MAPPINGS['Kf'],
    FEATURE_MAPPINGS['Ks'],
    FEATURE_MAPPINGS['AL/CR'],
    FEATURE_MAPPINGS['ACD'],
]
column_mapping = {

}
# 设置输出路径
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

def load_data(file_path):
    """加载原始数据"""
    print(f"加载数据: {file_path}")
    
    try:
        df = pd.read_excel(file_path)
        print(f"数据加载成功，形状: {df.shape}")
        return df
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None
    
def preprocess_data(df, features):
    """预处理数据"""
    # 选择特征和目标变量
    dfc = df[features].copy()
    return dfc

def double_data(df):
    dfd = preprocess_data(df, DFEATURES)
    print(dfd.head())
    dfs = preprocess_data(df, SFEATURES)
    print(dfs.head())
    new_columns = [
        'Age 年龄', 'Gender 性别', '是否矫正',
        'Uncorrected visual acuity 裸眼视力', 'Non-cycloplegic SD 电脑验光球镜',
        'Cycloplegic SER 等效球镜', '筛查区域', 'AL 眼轴', 'Kf',
        'Ks', 'AL/CR 轴率比', 'ACD 前房深',
    ]
    # 为左右眼数据分别重命名列
    dfd_renamed = dfd.rename(columns=dict(zip(dfd.columns, new_columns)))
    dfs_renamed = dfs.rename(columns=dict(zip(dfs.columns, new_columns)))
    combined_df = pd.concat([dfd_renamed, dfs_renamed], ignore_index=True)
    
    file_path = './data/doubledata.xlsx'
    combined_df.to_excel(file_path, index=False)

def main():
    """主函数"""
    print("开始数据集划分...")
    # 设置输入文件路径
    input_file = os.path.join(DATA_DIR, '筛查预测原始数据.xlsx')
    
    df = load_data(input_file)
    double_data(df)
    
    


if __name__ == "__main__":
    main()
    


# 原
# 训练集大小: 9359 样本
# 验证集大小: 2340 样本


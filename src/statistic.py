import pandas as pd
import numpy as np

def analyze_categorical_data(file_path):
    """ 分析Excel文件中分类属性的值分布
    Parameters: file_path (str): Excel文件路径
    """
    try:
        # 读取Excel文件
        df = pd.read_excel(file_path)
        print("✅ 文件读取成功")
        print(f"数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}\n")
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return
    
    # 检查目标列是否存在
    target_columns = ['Wearing refractive correction是否屈光矫正', '筛查区域 1=城区；2=郊县']
    
    for col in target_columns:
        if col not in df.columns:
            print(f"⚠️ 列 '{col}' 不存在于数据中")
            print(f"可用的列: {list(df.columns)}")
            # 尝试查找相似的列名
            similar_cols = [c for c in df.columns if col.split()[0] in c]
            if similar_cols:
                print(f"相似的列名: {similar_cols}")
            continue
        print(f"📊 '{col}' 的值分布统计:")
        print("-" * 50)
        # 统计每个值的数量
        value_counts = df[col].value_counts().sort_index()
        total_count = value_counts.sum()
        # 打印统计结果
        for value, count in value_counts.items():
            percentage = (count / total_count) * 100
            print(f"值 {value}: {count} 个 ({percentage:.2f}%)")
        
        missing_count = df[col].isnull().sum() # 统计缺失值
        if missing_count > 0:
            missing_percentage = (missing_count / len(df)) * 100
            print(f"缺失值: {missing_count} 个 ({missing_percentage:.2f}%)")
        print(f"总计: {total_count} 个有效值")
        # 获取值的具体含义（如果列名中有说明）
        if '1=城区；2=郊县' in col:
            print("\n📝 值含义说明:")
            print("1: 城区")
            print("2: 郊县")
        elif '是否屈光矫正' in col:
            print("\n📝 值含义说明:")
            print("请检查数据中该列的具体编码规则")
        print("\n" + "="*60 + "\n")

def calculate_eye_metrics_averages(file_path):
    """ 计算眼睛相关指标的平均值
    Parameters: file_path (str): Excel文件路径
    Returns: dict: 包含各指标平均值的字典
    """
    try:
        # 读取Excel文件
        df = pd.read_excel(file_path)
        print("✅ 文件读取成功")
        print(f"数据形状: {df.shape}")
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return None
    
    # 定义要计算的列
    target_columns = [
        'Uncorrected visual acuity_OD 裸眼视力右',
        'Uncorrected visual acuity_OS 裸眼视力左',
        'AL_OD右眼眼轴',
        'AL_OS左眼眼轴',
        'ACD_OD右眼前房深',
        'ACD_OS左眼前房深',
        'AL/CR_OD轴率比右',
        'AL/CR_OS轴率比左',
        'Kf_OD',
        'Kf_OS',
        'Ks_OD',
        'Ks_OS'
    ]
    
    # 检查哪些列存在
    existing_columns = [col for col in target_columns if col in df.columns]
    missing_columns = [col for col in target_columns if col not in df.columns]
    
    if missing_columns:
        print(f"⚠️ 以下列不存在: {missing_columns}")
        print(f"📊 将计算存在的列: {existing_columns}")
    if not existing_columns:
        print("❌ 没有找到任何目标列")
        return None
    # 计算平均值
    results = {}
    print("\n" + "="*60)
    print("📊 眼睛指标平均值统计")
    print("="*60)
    
    for col in existing_columns:
        # 过滤掉无效值（NaN、无穷大等）
        valid_data = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid_data) > 0:
            mean_value = valid_data.mean()
            std_value = valid_data.std()
            count = len(valid_data)
            missing_count = df[col].isnull().sum()
            
            results[col] = {
                'mean': mean_value,
                'std': std_value,
                'count': count,
                'missing': missing_count
            }
            print(f"\n{col}:")
            print(f"  平均值: {mean_value:.4f}")
            print(f"  标准差: {std_value:.4f}")
            print(f"  有效样本数: {count}")
            print(f"  缺失值数: {missing_count}")
            # 显示一些基本统计信息
            print(f"  最小值: {valid_data.min():.4f}")
            print(f"  最大值: {valid_data.max():.4f}")
        else:
            print(f"\n{col}: 无有效数据")
            results[col] = {'mean': np.nan, 'std': np.nan, 'count': 0, 'missing': len(df)}
    # 计算左右眼的差异
    print("\n" + "="*60)
    print("👁️ 左右眼差异分析")
    print("="*60)
    # 定义左右眼配对
    eye_pairs = [
        ('Uncorrected visual acuity_OD 裸眼视力右', 'Uncorrected visual acuity_OS 裸眼视力左', '裸眼视力'),
        ('AL_OD右眼眼轴', 'AL_OS左眼眼轴', '眼轴长度'),
        ('ACD_OD右眼前房深', 'ACD_OS左眼前房深', '前房深度'),
        ('AL/CR_OD轴率比右', 'AL/CR_OS轴率比左', '轴率比'),
        ('Kf_OD', 'Kf_OS', '平坦角膜曲率'),
        ('Ks_OD', 'Ks_OS', '陡峭角膜曲率')
    ]
    for od_col, os_col, metric_name in eye_pairs:
        if od_col in existing_columns and os_col in existing_columns:
            od_data = df[od_col].replace([np.inf, -np.inf], np.nan).dropna()
            os_data = df[os_col].replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(od_data) > 0 and len(os_data) > 0:
                # 只计算同时有左右眼数据的样本
                paired_data = df[[od_col, os_col]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(paired_data) > 0:
                    diff = paired_data[od_col] - paired_data[os_col]
                    mean_diff = diff.mean()
                    abs_mean_diff = diff.abs().mean()

                    print(f"\n{metric_name}左右眼差异:")
                    print(f"  平均差异(右-左): {mean_diff:.4f}")
                    print(f"  平均绝对差异: {abs_mean_diff:.4f}")
                    print(f"  配对样本数: {len(paired_data)}")
    return results


# 读取xlsx文件
df = pd.read_excel('data/筛查预测原始数据.xlsx')  # 可以指定工作表
# 'results/validation_dataset.xlsx''results/train_dataset.xlsx'
# 查看数据基本信息
print("数据形状:", df.shape)
print("\n前5行数据:")
#暂不print(df.head())

print("\n数据信息:")
#暂不print(df.info())

print("\n描述性统计:")
#暂不print(df.describe())

print("\nAge年龄统计:")
age_data = df['Age年龄'].dropna()
total_count = len(age_data)


# 查看列名
print("\n列名:", df.columns.tolist())
print(f"总样本数: {total_count}")
print(f"最大年龄: {age_data.max()}岁")
print(f"最小年龄: {age_data.min()}岁")
print(f"平均年龄: {age_data.mean():.2f}岁")
print(f"年龄中位数: {age_data.median():.2f}岁")
print("\n" + "=" * 40)
print("📈 按岁统计年龄分布")
print("=" * 40)
age_counts = age_data.value_counts().sort_index()
for age, count in age_counts.items():
    percentage = (count / total_count) * 100
    print(f"年龄 {age}岁: {count}人 ({percentage:.2f}%)")
# 按年龄段统计
print("\n" + "=" * 40)


# datapath = 'results/validation_dataset.xlsx'
# # 'results/validation_dataset.xlsx''results/train_dataset.xlsx'
# analyze_categorical_data(datapath)
# calculate_eye_metrics_averages(datapath)
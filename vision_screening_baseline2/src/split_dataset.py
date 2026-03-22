import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime
from config import FEATURE_MAPPING, AGE_GROUPS, SER_GROUPS, SPLIT_PARAMS

# 设置输出路径
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


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


def preprocess_data(df):
    """数据预处理"""
    print(f"原始数据形状: {df.shape}")

    # 检查缺失值
    print(f"缺失值统计:\n{df.isnull().sum()}")

    # 基本数据清洗
    # 1. 删除完全重复的行
    df = df.drop_duplicates()
    print(f"删除重复行后: {df.shape}")

    # 2. 处理年龄数据
    age_col = FEATURE_MAPPING.get('Age')
    if age_col in df.columns:
        # 确保年龄是数值型
        df[age_col] = pd.to_numeric(df[age_col], errors='coerce')
        # 删除年龄为空或异常的记录 >或年龄不满足实验要求的记录
        initial_count = len(df)
        df = df.dropna(subset=[age_col])
        df = df[(df[age_col] >= 5) & (df[age_col] <= 18)]
        removed_count = initial_count - len(df)
        print(f"年龄数据处理后: {df.shape} (移除{removed_count}行)")
    # dfage4 = df.loc[df[age_col]==5]
    # print(dfage4)

    # 3. 处理屈光度数据 - 统一删除缺失值
    for feature, col in FEATURE_MAPPING.items():
        if feature in ['Non-cycloplegic SER', 'Cycloplegic SER'] and col in df.columns:
            # 确保屈光度是数值型
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # 删除缺失值
            initial_count = len(df)
            df = df.dropna(subset=[col])
            removed_count = initial_count - len(df)
            print(f"{feature}数据处理后: {df.shape} (移除{removed_count}行)")

    print(f"最终数据形状: {df.shape}")
    return df


def split_dataset(df, test_size=None, random_state=None):
    """划分训练集和验证集"""
    # 使用配置文件中的参数
    test_size = test_size or SPLIT_PARAMS.get('test_size', 0.2)
    random_state = random_state or SPLIT_PARAMS.get('random_state', 42)

    # 根据年龄分层抽样
    age_col = FEATURE_MAPPING.get('Age')
    if age_col in df.columns:
        # 创建年龄分组
        bins = [group[0] for group in AGE_GROUPS.values()]
        bins.append(AGE_GROUPS[list(AGE_GROUPS.keys())[-1]][1] + 1)  # 添加最后一个上限
        bins[0] -= 1
        print(f'bins: {bins}')

        df['Age_Group'] = pd.cut(df[age_col],
                                 bins=bins,
                                 labels=list(AGE_GROUPS.keys()))
        # 应当注意到cut对左界也是开区间，
        # 也就是说cut之后年龄为5的（4个）样本均未分到正确的组中
        # bins[0] -= 1 和 如下代码都可以直接解决这一问题
        # print(f'分层前数据个数：{df.shape[0]}')
        # df = df.dropna(subset=['Age_Group'])
        # print(f'分层后数据个数：{df.shape[0]}')

        # 分层抽样
        train_df, val_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['Age_Group']
        )
    else:
        # 如果没有年龄列，则进行随机抽样
        train_df, val_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state
        )

    print(f"训练集大小: {train_df.shape}")
    print(f"验证集大小: {val_df.shape}")

    return train_df, val_df


def generate_statistics(train_df, val_df):
    """生成数据集统计信息"""
    stats = {}

    # 1. 样本数量
    stats['train_samples'] = len(train_df)
    stats['val_samples'] = len(val_df)
    print(f"训练集样本数: {stats['train_samples']}")
    print(f"验证集样本数: {stats['val_samples']}")

    # 2. 年龄统计
    age_col = FEATURE_MAPPING.get('Age')
    if age_col in train_df.columns:
        # 训练集年龄统计
        train_age_non_missing = train_df[age_col].dropna()
        stats['train_age_mean'] = train_age_non_missing.mean()
        stats['train_age_std'] = train_age_non_missing.std()
        stats['train_age_non_missing'] = len(train_age_non_missing)

        # 验证集年龄统计
        val_age_non_missing = val_df[age_col].dropna()
        stats['val_age_mean'] = val_age_non_missing.mean()
        stats['val_age_std'] = val_age_non_missing.std()
        stats['val_age_non_missing'] = len(val_age_non_missing)

        print(f"训练集年龄非缺失值: {stats['train_age_non_missing']}")
        print(f"验证集年龄非缺失值: {stats['val_age_non_missing']}")

        # 年龄分组统计 - 使用非缺失值计算百分比
        for group_name, (min_age, max_age) in AGE_GROUPS.items():
            # 训练集
            group_count = len(train_df[(train_df[age_col] >= min_age) & (train_df[age_col] <= max_age)])
            group_percent = group_count / stats['train_age_non_missing'] * 100 if stats[
                                                                                      'train_age_non_missing'] > 0 else 0
            stats[f'train_age_{group_name}_count'] = group_count
            stats[f'train_age_{group_name}_percent'] = group_percent

            # 验证集
            group_count = len(val_df[(val_df[age_col] >= min_age) & (val_df[age_col] <= max_age)])
            group_percent = group_count / stats['val_age_non_missing'] * 100 if stats['val_age_non_missing'] > 0 else 0
            stats[f'val_age_{group_name}_count'] = group_count
            stats[f'val_age_{group_name}_percent'] = group_percent

    # 3. 屈光度统计（只保留一次）
    for ser_type in ['Non-cycloplegic SER', 'Cycloplegic SER']:
        ser_col = FEATURE_MAPPING.get(ser_type)
        if ser_col and ser_col in train_df.columns:
            # 非缺失值数量
            train_ser_non_missing = train_df[ser_col].dropna()
            val_ser_non_missing = val_df[ser_col].dropna()

            stats[f'train_{ser_type}_non_missing'] = len(train_ser_non_missing)
            stats[f'val_{ser_type}_non_missing'] = len(val_ser_non_missing)

            print(f"训练集{ser_type}非缺失值: {stats[f'train_{ser_type}_non_missing']}")
            print(f"验证集{ser_type}非缺失值: {stats[f'val_{ser_type}_non_missing']}")

            # 训练集屈光度统计
            stats[f'train_{ser_type}_mean'] = train_ser_non_missing.mean()
            stats[f'train_{ser_type}_std'] = train_ser_non_missing.std()

            # 验证集屈光度统计
            stats[f'val_{ser_type}_mean'] = val_ser_non_missing.mean()
            stats[f'val_{ser_type}_std'] = val_ser_non_missing.std()

            # 屈光度分组统计 - 使用非缺失值计算百分比
            for group_name, (min_ser, max_ser) in SER_GROUPS.items():
                # 训练集
                group_count = len(train_df[(train_df[ser_col] > min_ser) & (train_df[ser_col] <= max_ser)])
                group_percent = group_count / stats[f'train_{ser_type}_non_missing'] * 100 if stats[
                                                                                                  f'train_{ser_type}_non_missing'] > 0 else 0
                stats[f'train_{ser_type}_{group_name}_count'] = group_count
                stats[f'train_{ser_type}_{group_name}_percent'] = group_percent

                # 验证集
                group_count = len(val_df[(val_df[ser_col] > min_ser) & (val_df[ser_col] <= max_ser)])
                group_percent = group_count / stats[f'val_{ser_type}_non_missing'] * 100 if stats[
                                                                                                f'val_{ser_type}_non_missing'] > 0 else 0
                stats[f'val_{ser_type}_{group_name}_count'] = group_count
                stats[f'val_{ser_type}_{group_name}_percent'] = group_percent

    # 4. 性别统计
    gender_col = FEATURE_MAPPING.get('Gender')
    if gender_col and gender_col in train_df.columns:
        # 训练集性别统计
        train_gender_counts = train_df[gender_col].value_counts()
        stats['train_gender_male'] = train_gender_counts.get('M', 0)
        stats['train_gender_female'] = train_gender_counts.get('F', 0)

        # 验证集性别统计
        val_gender_counts = val_df[gender_col].value_counts()
        stats['val_gender_male'] = val_gender_counts.get('M', 0)
        stats['val_gender_female'] = val_gender_counts.get('F', 0)

    # 5. 其他眼部特征统计
    for feature, col in FEATURE_MAPPING.items():
        if feature in ['AL', 'Kf', 'Ks', 'AL/CR', 'ACD'] and col in train_df.columns:
            # 训练集统计
            train_non_missing = train_df[col].dropna()
            stats[f'train_{feature}_mean'] = train_non_missing.mean()
            stats[f'train_{feature}_std'] = train_non_missing.std()
            stats[f'train_{feature}_non_missing'] = len(train_non_missing)

            # 验证集统计
            val_non_missing = val_df[col].dropna()
            stats[f'val_{feature}_mean'] = val_non_missing.mean()
            stats[f'val_{feature}_std'] = val_non_missing.std()
            stats[f'val_{feature}_non_missing'] = len(val_non_missing)

    return stats


def save_datasets(train_df, val_df, stats, output_dir):
    """保存结果"""
    # 保存划分后的数据集
    train_df.to_excel(os.path.join(output_dir, 'train_dataset.xlsx'), index=False)
    val_df.to_excel(os.path.join(output_dir, 'validation_dataset.xlsx'), index=False)

    # 保存统计信息
    stats_df = pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])
    stats_df.to_excel(os.path.join(output_dir, 'dataset_statistics.xlsx'), index=False)

    # 生成统计报告
    with open(os.path.join(output_dir, 'dataset_report.txt'), 'w', encoding='utf-8') as f:
        f.write("数据集划分报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 1. 样本数量
        f.write("1. 样本数量\n")
        f.write("-" * 30 + "\n")
        f.write(f"训练集: {stats['train_samples']} 样本\n")
        f.write(f"验证集: {stats['val_samples']} 样本\n\n")

        # 2. 年龄分布
        if 'train_age_mean' in stats:
            f.write("2. 年龄分布\n")
            f.write("-" * 30 + "\n")
            f.write(f"训练集: 平均 {stats['train_age_mean']:.2f} ± {stats['train_age_std']:.2f} 岁\n")
            f.write(f"验证集: 平均 {stats['val_age_mean']:.2f} ± {stats['val_age_std']:.2f} 岁\n\n")

            # 年龄分组
            f.write("年龄分组统计:\n")
            for group_name in AGE_GROUPS.keys():
                train_count = stats.get(f'train_age_{group_name}_count', 0)
                train_percent = stats.get(f'train_age_{group_name}_percent', 0)
                val_count = stats.get(f'val_age_{group_name}_count', 0)
                val_percent = stats.get(f'val_age_{group_name}_percent', 0)

                f.write(f"  {group_name}岁:\n")
                f.write(f"    训练集: {train_count} 样本 ({train_percent:.2f}%)\n")
                f.write(f"    验证集: {val_count} 样本 ({val_percent:.2f}%)\n")
            f.write("\n")

        # 3. 屈光度分布
        for ser_type in ['Non-cycloplegic SER', 'Cycloplegic SER']:
            if f'train_{ser_type}_mean' in stats:
                f.write(f"3. {ser_type}分布\n")
                f.write("-" * 30 + "\n")
                f.write(
                    f"训练集: 平均 {stats[f'train_{ser_type}_mean']:.2f} ± {stats[f'train_{ser_type}_std']:.2f} D\n")
                f.write(f"验证集: 平均 {stats[f'val_{ser_type}_mean']:.2f} ± {stats[f'val_{ser_type}_std']:.2f} D\n\n")

                # 屈光度分组
                f.write(f"{ser_type}分组统计:\n")
                for group_name in SER_GROUPS.keys():
                    train_count = stats.get(f'train_{ser_type}_{group_name}_count', 0)
                    train_percent = stats.get(f'train_{ser_type}_{group_name}_percent', 0)
                    val_count = stats.get(f'val_{ser_type}_{group_name}_count', 0)
                    val_percent = stats.get(f'val_{ser_type}_{group_name}_percent', 0)

                    f.write(f"  {group_name} D:\n")
                    f.write(f"    训练集: {train_count} 样本 ({train_percent:.2f}%)\n")
                    f.write(f"    验证集: {val_count} 样本 ({val_percent:.2f}%)\n")
                f.write("\n")

        # 4. 性别分布
        if 'train_gender_male' in stats:
            f.write("4. 性别分布\n")
            f.write("-" * 30 + "\n")
            f.write(f"男性:\n")
            f.write(f"  训练集: {stats['train_gender_male']} 样本\n")
            f.write(f"  验证集: {stats['val_gender_male']} 样本\n\n")

            f.write("女性:\n")
            f.write(f"  训练集: {stats['train_gender_female']} 样本\n")
            f.write(f"  验证集: {stats['val_gender_female']} 样本\n\n")

        # 5. 其他眼部特征
        f.write("5. 其他眼部特征统计\n")
        f.write("-" * 30 + "\n")

        for feature in ['AL', 'Kf', 'Ks', 'AL/CR', 'ACD']:
            if f'train_{feature}_mean' in stats:
                f.write(f"{feature}:\n")
                f.write(f"   训练集: 平均 {stats[f'train_{feature}_mean']:.2f} ± {stats[f'train_{feature}_std']:.2f}\n")
                f.write(f"   验证集: 平均 {stats[f'val_{feature}_mean']:.2f} ± {stats[f'val_{feature}_std']:.2f}\n\n")

    print(f"结果已保存到 {output_dir} 目录")


def main():
    """主函数"""
    print("开始数据集划分...")

    # 设置输入文件路径
    input_file = os.path.join(DATA_DIR, 'doubledata.xlsx')

    # 1. 加载数据
    df = load_data(input_file)
    if df is None:
        print("无法加载数据，程序退出")
        return

    # 2. 数据预处理
    df = preprocess_data(df)

    # 3. 划分数据集
    train_df, val_df = split_dataset(df)

    # 4. 生成统计信息
    stats = generate_statistics(train_df, val_df)

    # 5. 保存数据集和统计信息
    save_datasets(train_df, val_df, stats, RESULTS_DIR)

    print(f"数据集划分完成，结果保存在 {RESULTS_DIR} 目录下")
    print(f"训练集大小: {train_df.shape[0]} 样本")
    print(f"验证集大小: {val_df.shape[0]} 样本")


if __name__ == "__main__":
    main()
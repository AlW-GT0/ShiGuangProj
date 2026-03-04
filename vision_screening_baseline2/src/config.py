#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置文件，用于映射表格属性名与特征名
"""

# 特征名映射
FEATURE_MAPPING = {
    # 年龄映射
    'Age': 'Age 年龄',  # 可能的列名: '年龄', 'Age', '患者年龄' 等
    # 性别映射
    'Gender': 'Gender 性别',  # 可能的列名: '性别', 'Gender', '患者性别' 等
    #是否戴镜映射
    'Wearing refractive correction': '是否矫正',
    #裸眼映射
    'Uncorrected viosual acuity': 'Uncorrected visual acuity 裸眼视力',
    # 屈光度映射
    'Non-cycloplegic SER': 'Non-cycloplegic SD 电脑验光球镜',  # 非散瞳等效球镜度数(右眼)
    'Cycloplegic SER': 'Cycloplegic SER 等效球镜',  # 散瞳等效球镜度数(右眼)
    # 环境映射
    'District': '筛查区域',  # 可能的列名: '环境', 'Environment', '测试环境' 等’
    # 眼轴长度映射
    'AL': 'AL 眼轴',  # 眼轴长度(右眼)
    # 角膜曲率映射
    'Kf': 'Kf',  # 角膜曲率(右眼)
    'Ks': 'Ks',  # 角膜曲率(右眼)
    # 眼轴角膜曲率比映射
    'AL/CR': 'AL/CR 轴率比',  # 眼轴角膜曲率比(右眼)
    # 前房深度映射
    'ACD': 'ACD 前房深',  # 前房深度(右眼)
}
FEATURE_MAPPINGD = {
    # 年龄映射
    'Age': 'Age年龄',  # 可能的列名: '年龄', 'Age', '患者年龄' 等
    # 性别映射
    'Gender': 'Gender性别',  # 可能的列名: '性别', 'Gender', '患者性别' 等
    #是否戴镜映射
    'Wearing refractive correction': '是否矫正1=不戴镜；2=框架或隐性眼镜矫正',
    #裸眼映射
    'Uncorrected viosual acuity': 'Uncorrected visual acuity_OD 裸眼视力右',
    # 屈光度映射
    'Non-cycloplegic SER': 'Non-cycloplegic SD_OD 电脑验光球镜右',  # 非散瞳等效球镜度数(右眼)
    'Cycloplegic SER': 'Cycloplegic SER_OD右眼等效球镜',  # 散瞳等效球镜度数(右眼)
    # 环境映射
    'District': '筛查区域 1=城区；2=郊县',  # 可能的列名: '环境', 'Environment', '测试环境' 等’
    # 眼轴长度映射
    'AL': 'AL_OD右眼眼轴',  # 眼轴长度(右眼)
    # 角膜曲率映射
    'Kf': 'Kf_OD',  # 角膜曲率(右眼)
    'Ks': 'Ks_OD',  # 角膜曲率(右眼)
    # 眼轴角膜曲率比映射
    'AL/CR': 'AL/CR_OD轴率比右',  # 眼轴角膜曲率比(右眼)
    # 前房深度映射
    'ACD': 'ACD_OD右眼前房深',  # 前房深度(右眼)
}
FEATURE_MAPPINGS = {
    # 年龄映射
    'Age': 'Age年龄',  # 可能的列名: '年龄', 'Age', '患者年龄' 等
    # 性别映射
    'Gender': 'Gender性别',  # 可能的列名: '性别', 'Gender', '患者性别' 等
    #是否戴镜映射
    'Wearing refractive correction': '是否矫正1=不戴镜；2=框架或隐性眼镜矫正',
    #裸眼映射
    'Uncorrected viosual acuity': 'Uncorrected visual acuity_OS 裸眼视力左',
    # 屈光度映射
    'Non-cycloplegic SER': 'Non-cycloplegic SD_OS电脑验光球镜左',  # 非散瞳等效球镜度数(左眼)
    'Cycloplegic SER': 'Cycloplegic SER_OS左眼等效球镜',  # 散瞳等效球镜度数(左眼)
    # 环境映射
    'District': '筛查区域 1=城区；2=郊县',  # 可能的列名: '环境', 'Environment', '测试环境' 等’
    # 眼轴长度映射
    'AL': 'AL_OS左眼眼轴',  # 眼轴长度(左眼)
    # 角膜曲率映射
    'Kf': 'Kf_OS',  # 角膜曲率(左眼)
    'Ks': 'Ks_OS',  # 角膜曲率(左眼)
    # 眼轴角膜曲率比映射
    'AL/CR': 'AL/CR_OS轴率比左',  # 眼轴角膜曲率比(左眼)
    # 前房深度映射
    'ACD': 'ACD_OS左眼前房深',  # 前房深度(左眼)
}

# 年龄分组
AGE_GROUPS = {
    '<=4': (0, 4),
    '5-11': (5, 11),
    '12-15': (12, 15),
    '16-18': (16, 18),
    '>18': (19, 100)
}

# 屈光度分组
SER_GROUPS = {
    '<=−6.00D': (float('-inf'), -6.0),
    '>−6.00D,<=−3.00D': (-6.0, -3.0),
    '>−3.00D,<=−0.50D': (-3.0, -0.5),
    '>−0.50,<=0.50D': (-0.5, 0.5),
    '>0.50,<=3.00D': (0.5, 3.0),
    '>3.00D': (3.0, float('inf'))
}

# 数据集划分参数
SPLIT_PARAMS = {
    'test_size': 0.2,  # 验证集比例
    'random_state': 42  # 随机种子
}
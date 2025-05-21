#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
移除包含缺失值的行

此脚本用于清理 aki_hypertension_data_processed.csv 文件中的缺失值。
会直接修改原文件。
"""

import pandas as pd
from pathlib import Path
from datetime import datetime


def clean_data():
    """
    清理数据文件中的缺失值
    """
    # 文件路径
    file_path = Path('data/aki_hypertension_data_processed.csv')
    
    # 记录开始时间
    start_time = datetime.now()
    print(f"[{start_time.strftime('%Y-%m-%d %H:%M:%S')}] 开始处理文件: {file_path}")
    
    # 读取CSV文件
    print("正在读取数据...")
    df = pd.read_csv(file_path)
    print(f"原始数据大小: {df.shape[0]} 行 × {df.shape[1]} 列")
    
    # 检查缺失值
    missing_before = df.isnull().sum().sum()
    if missing_before == 0:
        print("\n警告: 数据中未发现缺失值，无需处理")
        return
    
    # 统计每列的缺失值数量
    print("\n各列缺失值统计:")
    missing_cols = df.isnull().sum()
    missing_cols = missing_cols[missing_cols > 0]
    for col, count in missing_cols.items():
        print(f"  - {col}: {count} 个缺失值 ({(count/len(df))*100:.2f}%)")
    
    # 移除包含缺失值的行
    print("\n正在移除包含缺失值的行...")
    df_cleaned = df.dropna()
    
    # 计算移除的行数
    n_removed = len(df) - len(df_cleaned)
    
    # 记录结束时间
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # 输出统计信息
    print(f"\n[{end_time.strftime('%Y-%m-%d %H:%M:%S')}] 处理完成! 耗时: {duration:.2f}秒")
    print(f"原始数据行数: {len(df)}")
    print(f"移除了 {n_removed} 行包含缺失值的数据 ({(n_removed/len(df))*100:.2f}%)")
    print(f"处理后数据行数: {len(df_cleaned)}")
    
    # 保存处理后的数据（覆盖原文件）
    df_cleaned.to_csv(file_path, index=False)
    print(f"\n处理后的数据已保存到: {file_path}")
    
    # 检查是否还有缺失值
    if df_cleaned.isnull().sum().sum() > 0:
        print("警告: 处理后数据中仍然存在缺失值")
    else:
        print("所有缺失值已成功移除")


if __name__ == "__main__":
    clean_data()

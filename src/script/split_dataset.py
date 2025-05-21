import pandas as pd
import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split

# 安装imblearn库: pip install imbalanced-learn
try:
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks, ClusterCentroids
    from imblearn.combine import SMOTETomek
    from collections import Counter
except ImportError:
    print("警告: 未找到imblearn库，请使用 'pip install imbalanced-learn' 安装")

def undersample_data(input_file, output_file, method='random', ratio=1.0, random_state=42):
    """
    对数据集进行欠采样处理
    
    参数:
    - input_file: 输入CSV文件路径
    - output_file: 输出CSV文件路径
    - method: 欠采样方法，可选'random'（随机欠采样）, 'tomek'（Tomek Links）, 
              'cluster'（聚类欠采样）, 'smote_tomek'（SMOTE+Tomek）
    - ratio: 多数类与少数类的比例，默认为1.0（平衡）
    - random_state: 随机种子
    """
    # 读取数据
    print(f"读取数据: {input_file}")
    data = pd.read_csv(input_file)
    
    # 提取特征和标签
    # 注意：这里简化处理，实际应用中可能需要更复杂的特征提取
    static_cols = ['gender', 'age']
    
    # 动态特征（仅用于示例，实际应根据具体情况选择）
    dynamic_cols = [col for col in data.columns if 'creatinine_t' in col or 'urine_output_t' in col]
    
    # 选择前几个时间步的特征，避免维度过高
    selected_dynamic_cols = dynamic_cols[:10]  
    
    # 合并特征
    X_cols = static_cols + selected_dynamic_cols
    X = data[X_cols].fillna(0)  # 简单处理缺失值
    
    # 将AKI时间转换为二分类标签（-1=未发生，其他=发生）
    y = (data['aki_time'] != -1).astype(int)
    
    # 打印原始类别分布
    print("原始类别分布:")
    print(Counter(y))
    
    # 选择欠采样方法
    if method == 'random':
        sampler = RandomUnderSampler(sampling_strategy={0: int(ratio * sum(y==1)), 1: sum(y==1)}, 
                                     random_state=random_state)
    elif method == 'tomek':
        sampler = TomekLinks()
    elif method == 'cluster':
        sampler = ClusterCentroids(sampling_strategy={0: int(ratio * sum(y==1)), 1: sum(y==1)},
                                   random_state=random_state)
    elif method == 'smote_tomek':
        sampler = SMOTETomek(sampling_strategy={0: int(ratio * sum(y==1)), 1: sum(y==1)},
                             random_state=random_state)
    else:
        raise ValueError(f"不支持的欠采样方法: {method}")
    
    # 执行欠采样
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    
    # 打印欠采样后的类别分布
    print("欠采样后的类别分布:")
    print(Counter(y_resampled))
    
    # 获取欠采样后的索引
    if hasattr(sampler, 'sample_indices_'):
        # 如果采样器有sample_indices_属性（如RandomUnderSampler）
        resampled_data = data.iloc[sampler.sample_indices_].copy()
    else:
        # 否则，创建一个新的DataFrame
        resampled_indices = []
        for i, val in enumerate(y):
            if val == 1:  # 对于少数类，保留所有样本
                resampled_indices.append(i)
        
        # 对于多数类，随机选择样本
        neg_indices = np.where(y == 0)[0]
        np.random.seed(random_state)
        selected_neg_indices = np.random.choice(neg_indices, size=int(ratio * sum(y==1)), replace=False)
        resampled_indices.extend(selected_neg_indices)
        
        # 按原始顺序排序
        resampled_indices = sorted(resampled_indices)
        resampled_data = data.iloc[resampled_indices].copy()
    
    # 保存欠采样后的数据
    resampled_data.to_csv(output_file, index=False)
    print(f"欠采样后的数据已保存至: {output_file}")
    
    return resampled_data

def split_dataset(input_file, train_ratio=0.8, random_state=42, output_dir='data', undersampling=False, undersampling_method='random', undersampling_ratio=1.0):
    """
    将数据集分割为训练集和评估集
    
    参数:
    - input_file: 输入CSV文件路径
    - train_ratio: 训练集比例，默认为0.8（80%训练，20%评估）
    - random_state: 随机种子，确保结果可重现
    - output_dir: 输出目录，默认为'data'
    """
    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取数据
        print(f"正在读取数据: {input_file}")
        data = pd.read_csv(input_file)
        print(f"原始数据集大小: {len(data)} 条记录")
        
        # 分割数据集
        train_data, eval_data = train_test_split(
            data,
            train_size=train_ratio,
            random_state=random_state,
            stratify=data['aki_time'].apply(lambda x: 1 if x > 0 else 0)  # 保持AKI分布一致
        )
        
        # 生成输出文件名
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        train_file = os.path.join(output_dir, f"{base_name}_train.csv")
        eval_file = os.path.join(output_dir, f"{base_name}_eval.csv")
        
        # 如果需要欠采样，对训练集进行处理
        if undersampling:
            print(f"对训练集进行{undersampling_method}欠采样，比例: {undersampling_ratio}")
            undersampled_train_file = os.path.join(output_dir, f"{base_name}_train_undersampled.csv")
            train_data = undersample_data(train_file, undersampled_train_file, 
                                        method=undersampling_method, 
                                        ratio=undersampling_ratio, 
                                        random_state=random_state)
            train_file = undersampled_train_file
        else:
            # 保存原始训练集
            train_data.to_csv(train_file, index=False)
        
        # 保存评估集
        eval_data.to_csv(eval_file, index=False)
        
        # 打印统计信息
        print("\n数据集分割完成:")
        print(f"训练集: {len(train_data)} 条记录 ({len(train_data)/len(data)*100:.1f}%)")
        print(f"评估集: {len(eval_data)} 条记录 ({len(eval_data)/len(data)*100:.1f}%)")
        print(f"\nAKI分布 (训练集):")
        print(train_data['aki_time'].value_counts().sort_index())
        print(f"\nAKI分布 (评估集):")
        print(eval_data['aki_time'].value_counts().sort_index())
        
        print(f"\n训练集已保存至: {train_file}")
        print(f"评估集已保存至: {eval_file}")
        
        return train_file, eval_file
    
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        return None, None

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='分割数据集并进行欠采样')
    parser.add_argument('--input', type=str, default='data/aki_hypertension_data_processed.csv', help='输入数据路径')
    parser.add_argument('--output_dir', type=str, default='data', help='输出目录')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例，默认为0.8')
    parser.add_argument('--random_state', type=int, default=42, help='随机种子')
    parser.add_argument('--undersample', action='store_true', help='是否对训练集进行欠采样')
    parser.add_argument('--method', type=str, default='random', choices=['random', 'tomek', 'cluster', 'smote_tomek'], help='欠采样方法')
    parser.add_argument('--ratio', type=float, default=1.0, help='多数类与少数类的比例，默认为1.0（平衡）')
    return parser.parse_args()

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    # 分割数据集
    train_file, eval_file = split_dataset(
        input_file=args.input,
        train_ratio=args.train_ratio,
        random_state=args.random_state,
        output_dir=args.output_dir,
        undersampling=args.undersample,
        undersampling_method=args.method,
        undersampling_ratio=args.ratio
    )
    
    if train_file and eval_file:
        print("\n数据集处理成功完成！")
        
        # 打印使用说明
        if args.undersample:
            print("训练命令示例:")
            base_name = os.path.splitext(os.path.basename(args.input))[0]
            print(f"  python src/train/train.py --data {args.output_dir}/{base_name}_train_undersampled.csv")
            
            print("评估命令示例:")
            print(f"  python src/eval/eval.py --data {args.output_dir}/{base_name}_eval.csv")
        else:
            print("训练命令示例:")
            base_name = os.path.splitext(os.path.basename(args.input))[0]
            print(f"  python src/train/train.py --data {args.output_dir}/{base_name}_train.csv")
            
            print("评估命令示例:")
            print(f"  python src/eval/eval.py --data {args.output_dir}/{base_name}_eval.csv")
    else:
        print("\n数据集处理失败，请检查错误信息。")
import sys
import os
import argparse
import torch
import numpy as np
sys.path.append("D:/05_Project/03_Python/RWKVSurv")

import pandas as pd
from torch.utils.data import random_split, TensorDataset, DataLoader



from src.model.model import AKIConfig, AKIPredictor, prepare_data
from src.train.trainer import Trainer, TrainerConfig

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练AKI预测模型')
    parser.add_argument('--data', type=str, default='data/generated_data_processed_train.csv', help='训练数据路径')
    parser.add_argument('--resume', action='store_true', help='是否从断点继续训练')
    parser.add_argument('--checkpoint', type=str, default='data/ckpt/trained-model-.pt', help='断点模型路径')
    parser.add_argument('--epochs', type=int, default=500, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128, help='批量大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--save_freq', type=int, default=10, help='模型保存频率')
    parser.add_argument('--save_path', type=str, default='data/ckpt/trained-model-', help='模型保存路径')
    # 模型相关参数
    parser.add_argument('--h', type=int, default=6, help='提前预测步数，小于这个时间步发生的数据先筛除')
    # 数据平衡相关参数
    parser.add_argument('--undersample', action='store_true', help='是否使用欠采样使正负样本比例为1:1')
    parser.add_argument('--weighted_loss', action='store_true', help='是否使用加权损失函数处理数据不平衡')
    parser.add_argument('--pos_weight', type=float, default=1.0, help='正样本权重，默认为7.0（多数类与少数类的比例）')
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 确保模型保存目录存在
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # 加载数据
    print(f'加载数据: {args.data}')
    data = pd.read_csv(args.data)
    
    # 准备数据
    print(f'提前预测步数 h = {args.h}')
    static_features, dynamic_features, targets, durations = prepare_data(data, h=args.h)
    
    # 如果使用欠采样，对数据进行重采样使正负样本比例为1:1
    if args.undersample:
        print("使用欠采样使正负样本比例为1:1...")
        
        # 获取正样本和负样本的索引
        pos_indices = torch.where(targets == 1)[0]
        neg_indices = torch.where(targets == 0)[0]
        
        # 计算正负样本数量
        n_pos = len(pos_indices)
        n_neg = len(neg_indices)
        
        print(f"原始数据: 正样本 {n_pos} 个, 负样本 {n_neg} 个, 正负比例 1:{n_neg/n_pos:.2f}")
        
        # 随机选择与正样本数量相同的负样本
        np.random.seed(42)  # 设置随机种子以确保可重复性
        selected_neg_indices = np.random.choice(neg_indices.numpy(), size=n_pos, replace=False)
        selected_neg_indices = torch.tensor(selected_neg_indices)
        
        # 合并选择的负样本和所有正样本的索引
        selected_indices = torch.cat([pos_indices, selected_neg_indices])
        
        # 根据选择的索引获取新的数据集
        static_features = static_features[selected_indices]
        dynamic_features = dynamic_features[selected_indices]
        targets = targets[selected_indices]
        durations = durations[selected_indices]
        
        # 验证欠采样后的数据集
        n_pos_after = torch.sum(targets == 1).item()
        n_neg_after = torch.sum(targets == 0).item()
        print(f"欠采样后: 正样本 {n_pos_after} 个, 负样本 {n_neg_after} 个, 正负比例 1:{n_neg_after/n_pos_after:.2f}")
        print(f"总样本数: {len(targets)} 个")
    
    # 设置总样本数和划分比例
    total_samples = len(static_features)
    train_size = int(0.8 * total_samples)
    test_size = total_samples - train_size

    # 随机划分数据集
    train_dataset, test_dataset = random_split(
        dataset=TensorDataset(static_features, dynamic_features, targets, durations),
        lengths=[train_size, test_size]
    )
    
    # 获取特征维度
    static_dim = static_features.shape[1]  # 静态特征维度
    dynamic_dim = dynamic_features.shape[2]  # 动态特征维度
    time_steps = dynamic_features.shape[1]  # 时间步数
    
    # 判断是否从断点继续训练
    if args.resume and os.path.exists(args.checkpoint):
        print(f'从断点继续训练: {args.checkpoint}')
        try:
            # 加载模型断点
            model = torch.load(args.checkpoint, map_location=torch.device('cpu'), weights_only=False)
            print('模型加载成功')
            
            # 确保模型与数据维度匹配
            if model.config.static_dim != static_dim or model.config.dynamic_dim != dynamic_dim or model.config.ctx_len != time_steps:
                print('警告: 断点模型的维度与当前数据不匹配')
                print(f'模型: static_dim={model.config.static_dim}, dynamic_dim={model.config.dynamic_dim}, ctx_len={model.config.ctx_len}')
                print(f'数据: static_dim={static_dim}, dynamic_dim={dynamic_dim}, ctx_len={time_steps}')
                print('将创建新模型')
                model = AKIPredictor(AKIConfig(
                    static_dim=static_dim,
                    dynamic_dim=dynamic_dim,
                    embed_dim=128,
                    n_layer=3,
                    n_head=4,
                    ctx_len=time_steps,
                    h=args.h
                ))
        except Exception as e:
            print(f'加载模型失败: {e}')
            print('将创建新模型')
            model = AKIPredictor(AKIConfig(
                static_dim=static_dim,
                dynamic_dim=dynamic_dim,
                embed_dim=128,
                n_layer=3,
                n_head=4,
                ctx_len=time_steps
            ))
    else:
        # 创建新模型
        print('创建新模型')
        model = AKIPredictor(AKIConfig(
            static_dim=static_dim,
            dynamic_dim=dynamic_dim,
            embed_dim=128,
            n_layer=3,
            n_head=4,
            ctx_len=time_steps,
            h=args.h
        ))
    
    # 训练配置
    train_config = TrainerConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epoch_save_frequency=args.save_freq,
        epoch_save_path=args.save_path,
        grad_norm_clip=1,  # 梯度范数裁剪阈值
    )
    
    print(f"\n训练配置:\n - 学习率: {train_config.learning_rate}\n - 批量大小: {train_config.batch_size}\n - 梯度范数裁剪阈值: {train_config.grad_norm_clip}\n")
    
    # 初始化训练器
    trainer = Trainer(
        model, 
        train_dataset, 
        test_dataset, 
        train_config, 
        use_weighted_loss=args.weighted_loss, 
        pos_weight=args.pos_weight
    )
    
    # 开始训练
    print('开始训练...')
    trainer.train()
    print('训练完成')

if __name__ == '__main__':
    main()
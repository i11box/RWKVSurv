import sys
import os
import argparse
import torch
import numpy as np
sys.path.append("D:/05_Project/03_Python/RWKVSurv")

import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

# 添加项目路径到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.model.model import AKIConfig, AKIPredictor, prepare_data
from src.model.ml_model import MLConfig, RandomForestModel, LogisticRegressionModel
from src.train.trainer import Trainer, TrainerConfig

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练AKI预测模型')
    parser.add_argument('--data', type=str, default='data/generated_data_processed_train.csv', help='训练数据路径')
    parser.add_argument('--resume', action='store_true', help='是否从断点继续训练')
    parser.add_argument('--checkpoint', type=str, default='data/ckpt/trained-model-.pt', help='断点模型路径')
    parser.add_argument('--epochs', type=int, default=1000, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128, help='批量大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--save_freq', type=int, default=10, help='模型保存频率')
    parser.add_argument('--save_path', type=str, default='data/ckpt/trained-model-', help='模型保存路径')
    # 模型相关参数
    parser.add_argument('--h', type=int, default=6, help='提前预测步数，小于这个时间步发生的数据先筛除')
    parser.add_argument('--model_type', type=str, default='RWKV', choices=['RWKV', 'LSTM', 'GRU', 'Transformer', 'S5', 'RandomForest', 'LogisticRegression'], help='模型类型: RWKV, LSTM, GRU, Transformer, S5, RandomForest, LogisticRegression')
    
    # LSTM特定参数
    parser.add_argument('--lstm_layers', type=int, default=3, help='LSTM层数')
    parser.add_argument('--lstm_bidirectional', action='store_true', help='是否使用双向LSTM')
    
    # GRU特定参数
    parser.add_argument('--gru_layers', type=int, default=3, help='GRU层数')
    parser.add_argument('--gru_bidirectional', action='store_true', help='是否使用双向GRU')
    
    # Transformer特定参数
    parser.add_argument('--attn_dropout', type=float, default=0.1, help='Transformer注意力机制的dropout率')
    parser.add_argument('--ff_activation', type=str, default='gelu', choices=['gelu', 'relu', 'silu', 'mish'], help='Transformer前馈网络的激活函数类型')
    
    # 随机森林特定参数
    parser.add_argument('--rf_n_estimators', type=int, default=100, help='随机森林中树的数量')
    parser.add_argument('--rf_max_depth', type=int, default=None, help='树的最大深度，None表示无限制')
    parser.add_argument('--rf_min_samples_split', type=int, default=2, help='分裂内部节点所需的最小样本数')
    parser.add_argument('--rf_min_samples_leaf', type=int, default=1, help='叶节点所需的最小样本数')
    
    # 逻辑回归特定参数
    parser.add_argument('--lr_C', type=float, default=1.0, help='逻辑回归正则化强度的倒数')
    parser.add_argument('--lr_penalty', type=str, default='l2', choices=['l1', 'l2', 'elasticnet', 'none'], help='逻辑回归正则化类型')
    parser.add_argument('--lr_solver', type=str, default='lbfgs', choices=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], help='逻辑回归优化算法')
    parser.add_argument('--lr_max_iter', type=int, default=100, help='逻辑回归最大迭代次数')
    
    # S5特定参数
    parser.add_argument('--s5_state_dim', type=int, default=None, help='S5状态空间维度，默认等于嵌入维度')
    parser.add_argument('--s5_bidir', action='store_true', help='是否使用双向S5')
    parser.add_argument('--s5_block_count', type=int, default=4, help='S5块数量')
    parser.add_argument('--s5_liquid', action='store_true', help='是否使用liquid S5')
    parser.add_argument('--s5_degree', type=int, default=1, help='S5度数')
    parser.add_argument('--s5_bc_init', type=str, default='dense', help='BC初始化方法')
    parser.add_argument('--s5_ff_mult', type=float, default=1.0, help='前馈网络乘数')
    parser.add_argument('--s5_glu', action='store_true', help='是否使用GLU')
    
    # 机器学习模型共用参数
    parser.add_argument('--ml_max_steps', type=int, default=None, help='机器学习模型使用的最大时间步数，None表示使用所有时间步')
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
                # 根据模型类型选择不同的模型类
                if args.model_type in ['RandomForest', 'LogisticRegression']:
                    # 使用机器学习模型
                    config = MLConfig(
                        static_dim=static_dim,
                        dynamic_dim=dynamic_dim,
                        h=args.h,
                        model_type=args.model_type,
                        # 随机森林特定参数
                        rf_n_estimators=args.rf_n_estimators,
                        rf_max_depth=args.rf_max_depth,
                        rf_min_samples_split=args.rf_min_samples_split,
                        rf_min_samples_leaf=args.rf_min_samples_leaf,
                        # 逻辑回归特定参数
                        lr_C=args.lr_C,
                        lr_penalty=args.lr_penalty,
                        lr_solver=args.lr_solver,
                        lr_max_iter=args.lr_max_iter,
                        # 机器学习模型共用参数
                        ml_max_steps=args.ml_max_steps
                    )
                    
                    if args.model_type == 'RandomForest':
                        model = RandomForestModel(config)
                        print(f"使用随机森林模型，树数量: {args.rf_n_estimators}, 最大深度: {args.rf_max_depth if args.rf_max_depth else '无限制'}")
                    else:  # LogisticRegression
                        model = LogisticRegressionModel(config)
                        print(f"使用逻辑回归模型，正则化强度: {args.lr_C}, 正则化类型: {args.lr_penalty}")
                    
                else:
                    # 使用深度学习模型
                    model = AKIPredictor(AKIConfig(
                        static_dim=static_dim,
                        dynamic_dim=dynamic_dim,
                        embed_dim=128,
                        n_layer=3,
                        n_head=4,
                        ctx_len=time_steps,
                        h=args.h,
                        model_type=args.model_type,
                        # LSTM特定参数
                        lstm_layers=args.lstm_layers,
                        lstm_bidirectional=args.lstm_bidirectional,
                        # GRU特定参数
                        gru_layers=args.gru_layers,
                        gru_bidirectional=args.gru_bidirectional,
                        # Transformer特定参数
                        attn_dropout=args.attn_dropout,
                        ff_activation=args.ff_activation,
                        
                        # S5特定参数
                        s5_state_dim=args.s5_state_dim,
                        s5_bidir=args.s5_bidir,
                        s5_block_count=args.s5_block_count,
                        s5_liquid=args.s5_liquid,
                        s5_degree=args.s5_degree,
                        s5_bc_init=args.s5_bc_init,
                        s5_ff_mult=args.s5_ff_mult,
                        s5_glu=args.s5_glu
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
                ctx_len=time_steps,
                h=args.h,
                model_type=args.model_type,
                # LSTM特定参数
                lstm_layers=args.lstm_layers,
                lstm_bidirectional=args.lstm_bidirectional,
                # GRU特定参数
                gru_layers=args.gru_layers,
                gru_bidirectional=args.gru_bidirectional,
                # Transformer特定参数
                attn_dropout=args.attn_dropout,
                ff_activation=args.ff_activation,
                
                # S5特定参数
                s5_state_dim=args.s5_state_dim,
                s5_bidir=args.s5_bidir,
                s5_block_count=args.s5_block_count,
                s5_liquid=args.s5_liquid,
                s5_degree=args.s5_degree,
                s5_bc_init=args.s5_bc_init,
                s5_ff_mult=args.s5_ff_mult,
                s5_glu=args.s5_glu
            ))
    else:
        # 创建新模型
        print('创建新模型')
        
        # 根据模型类型选择不同的模型类
        if args.model_type in ['RandomForest', 'LogisticRegression']:
            # 使用机器学习模型
            config = MLConfig(
                static_dim=static_dim,
                dynamic_dim=dynamic_dim,
                h=args.h,
                model_type=args.model_type,
                # 随机森林特定参数
                rf_n_estimators=args.rf_n_estimators,
                rf_max_depth=args.rf_max_depth,
                rf_min_samples_split=args.rf_min_samples_split,
                rf_min_samples_leaf=args.rf_min_samples_leaf,
                # 逻辑回归特定参数
                lr_C=args.lr_C,
                lr_penalty=args.lr_penalty,
                lr_solver=args.lr_solver,
                lr_max_iter=args.lr_max_iter,
                # 机器学习模型共用参数
                ml_max_steps=args.ml_max_steps
            )
            
            if args.model_type == 'RandomForest':
                model = RandomForestModel(config)
                print(f"使用随机森林模型，树数量: {args.rf_n_estimators}, 最大深度: {args.rf_max_depth if args.rf_max_depth else '无限制'}")
            else:  # LogisticRegression
                model = LogisticRegressionModel(config)
                print(f"使用逻辑回归模型，正则化强度: {args.lr_C}, 正则化类型: {args.lr_penalty}")
            # # 启用调试输出，保存到项目根目录下的debug_output目录
            # debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'debug_output')
            # model.set_debug(True, debug_dir=debug_dir)
            # print(f"调试模式已启用，调试输出将保存到: {debug_dir}")
        else:
            # 使用深度学习模型
            model = AKIPredictor(AKIConfig(
                static_dim=static_dim,
                dynamic_dim=dynamic_dim,
                embed_dim=128,
                n_layer=8,
                n_head=4,
                ctx_len=time_steps,
                h=args.h,
                model_type=args.model_type,
                # LSTM特定参数
                lstm_layers=args.lstm_layers,
                lstm_bidirectional=args.lstm_bidirectional,
                # GRU特定参数
                gru_layers=args.gru_layers,
                gru_bidirectional=args.gru_bidirectional,
                # Transformer特定参数
                attn_dropout=args.attn_dropout,
                ff_activation=args.ff_activation
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
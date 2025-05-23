import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import logging

logger = logging.getLogger(__name__)

class RandomForestModel:
    """
    随机森林模型，用于AKI预测
    
    该模型将时序数据转换为定长特征，然后使用随机森林进行分类
    """
    def __init__(self, config):
        self.config = config
        
        # 初始化随机森林分类器
        self.model = RandomForestClassifier(
            n_estimators=config.rf_n_estimators,
            max_depth=config.rf_max_depth,
            min_samples_split=config.rf_min_samples_split,
            min_samples_leaf=config.rf_min_samples_leaf,
            random_state=42
        )
        
        # 保存模型配置
        self.static_dim = config.static_dim
        self.dynamic_dim = config.dynamic_dim
        self.h = config.h
        
        logger.info(f"初始化随机森林模型: n_estimators={config.rf_n_estimators}, max_depth={config.rf_max_depth}")
    
    def prepare_flat_features(self, static_features, dynamic_features, max_steps=None):
        """
        将时序特征转换为定长特征
        
        参数:
        - static_features: 静态特征 [batch_size, static_dim]
        - dynamic_features: 动态特征 [batch_size, time_steps, dynamic_dim]
        - max_steps: 最大时间步数，如果为None则使用实际时间步数
        
        返回:
        - flat_features: 定长特征 [batch_size, static_dim + dynamic_dim * time_steps]
        """
        batch_size, time_steps, _ = dynamic_features.shape
        
        if max_steps is None:
            max_steps = time_steps
        elif max_steps > time_steps:
            # 如果指定的最大时间步数大于实际时间步数，则用第一步的数据补齐
            padding = np.repeat(dynamic_features[:, 0:1, :], max_steps - time_steps, axis=1)
            dynamic_features = np.concatenate([dynamic_features, padding], axis=1)
            time_steps = max_steps
        else:
            # 如果指定的最大时间步数小于实际时间步数，则截断
            dynamic_features = dynamic_features[:, :max_steps, :]
            time_steps = max_steps
        
        # 将动态特征展平
        flat_dynamic = dynamic_features.reshape(batch_size, -1)
        
        # 合并静态特征和展平的动态特征
        flat_features = np.concatenate([static_features, flat_dynamic], axis=1)
        
        return flat_features
    
    def fit(self, static_features, dynamic_features, targets):
        """
        训练随机森林模型
        
        参数:
        - static_features: 静态特征 [batch_size, static_dim]
        - dynamic_features: 动态特征 [batch_size, time_steps, dynamic_dim]
        - targets: 目标值，AKI是否发生 (0=未发生, 1=发生) [batch_size]
        
        返回:
        - self: 训练好的模型
        """
        # 将时序特征转换为定长特征
        flat_features = self.prepare_flat_features(static_features, dynamic_features)
        
        # 训练随机森林模型
        logger.info(f"训练随机森林模型，特征维度: {flat_features.shape}")
        self.model.fit(flat_features, targets)
        
        # 输出特征重要性
        feature_importances = self.model.feature_importances_
        logger.info(f"随机森林模型训练完成，特征重要性前10: {feature_importances.argsort()[-10:][::-1]}")
        
        return self
    
    def predict_proba(self, static_features, dynamic_features):
        """
        预测AKI发生概率
        
        参数:
        - static_features: 静态特征 [batch_size, static_dim]
        - dynamic_features: 动态特征 [batch_size, time_steps, dynamic_dim]
        
        返回:
        - probs: AKI发生概率 [batch_size]
        """
        # 将时序特征转换为定长特征
        flat_features = self.prepare_flat_features(static_features, dynamic_features)
        
        # 预测概率
        probs = self.model.predict_proba(flat_features)[:, 1]
        
        return probs
    
    def save(self, path):
        """
        保存模型
        
        参数:
        - path: 模型保存路径
        """
        # 创建目录
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存模型
        joblib.dump(self.model, path)
        logger.info(f"随机森林模型已保存到 {path}")
    
    def load(self, path):
        """
        加载模型
        
        参数:
        - path: 模型加载路径
        
        返回:
        - self: 加载好的模型
        """
        # 加载模型
        self.model = joblib.load(path)
        logger.info(f"随机森林模型已从 {path} 加载")
        
        return self
    
    def forward(self, static_features, dynamic_features, targets=None, durations=None, is_training=True):
        """
        前向传播 - 兼容AKIPredictor的接口
        
        参数:
        - static_features: 静态特征 [batch_size, static_dim]
        - dynamic_features: 动态特征 [batch_size, time_steps, dynamic_dim]
        - targets: 目标值，AKI是否发生 (0=未发生, 1=发生) [batch_size]
        - durations: AKI发生的时间点 [batch_size]
        - is_training: 是否处于训练模式
        
        返回:
        - aki_probs: 每个样本发生AKI的概率 [batch_size, 1]
        - loss: 损失值
        """
        # 转换为numpy数组
        static_np = static_features.cpu().numpy()
        dynamic_np = dynamic_features.cpu().numpy()
        
        # 预测概率
        probs = self.predict_proba(static_np, dynamic_np)
        
        # 转换为tensor
        import torch
        aki_probs = torch.tensor(probs, device=static_features.device).unsqueeze(-1)
        
        # 计算损失
        loss = None
        if targets is not None:
            import torch.nn.functional as F
            loss = F.binary_cross_entropy(aki_probs.squeeze(-1), targets)
        
        return aki_probs, loss

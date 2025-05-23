import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import os
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib

logger = logging.getLogger(__name__)

class MLConfig:
    """
    机器学习模型配置类
    
    用于配置随机森林和逻辑回归等机器学习模型的参数
    """
    def __init__(self, 
                 static_dim, 
                 dynamic_dim, 
                 h=6,                    # 提前预测步数，默认为6
                 model_type='RandomForest',  # 模型类型: 'RandomForest', 'LogisticRegression'
                 
                 # 随机森林特定参数
                 rf_n_estimators=100,     # 随机森林中树的数量
                 rf_max_depth=None,       # 树的最大深度
                 rf_min_samples_split=2,  # 分裂内部节点所需的最小样本数
                 rf_min_samples_leaf=1,   # 叶节点所需的最小样本数
                 
                 # 逻辑回归特定参数
                 lr_C=1.0,                # 正则化强度的倒数
                 lr_penalty='l2',         # 正则化类型
                 lr_solver='lbfgs',       # 优化算法
                 lr_max_iter=100,         # 最大迭代次数
                 
                 # 机器学习模型共用参数
                 ml_max_steps=None,       # 机器学习模型使用的最大时间步数，None表示使用所有时间步
                 
                 **kwargs):
        # 基本参数
        self.static_dim = static_dim      # 静态特征维度
        self.dynamic_dim = dynamic_dim    # 动态特征维度
        self.model_type = model_type      # 模型类型
        self.h = h                        # 提前预测步数
        
        # 随机森林特定参数
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_depth = rf_max_depth
        self.rf_min_samples_split = rf_min_samples_split
        self.rf_min_samples_leaf = rf_min_samples_leaf
        
        # 逻辑回归特定参数
        self.lr_C = lr_C
        self.lr_penalty = lr_penalty
        self.lr_solver = lr_solver
        self.lr_max_iter = lr_max_iter
        
        # 机器学习模型共用参数
        self.ml_max_steps = ml_max_steps
        
        # 处理其他参数
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        # 设置日志
        logger.info(f"初始化机器学习模型配置: 模型类型={self.model_type}")


class BaseMLModel:
    """
    机器学习模型基类
    
    提供共享的数据处理方法和接口
    """
    def __init__(self):
        """初始化基类，设置调试标志"""
        self._debug_enabled = False
        self._debug_dir = 'debug_output'
        self._debug_aki_saved = 0  # 已保存的AKI样本计数
        self._debug_non_aki_saved = 0  # 已保存的非AKI样本计数
        self._debug_max_samples = 5  # 每种类型最多保存的样本数
    
    def set_debug(self, enabled=True, debug_dir='debug_output', max_samples=5):
        """
        启用或禁用调试输出
        
        参数:
        - enabled: 是否启用调试输出
        - debug_dir: 调试输出目录
        - max_samples: 每种类型(AKI/非AKI)最多保存的样本数
        """
        self._debug_enabled = enabled
        self._debug_dir = debug_dir
        self._debug_aki_saved = 0
        self._debug_non_aki_saved = 0
        self._debug_max_samples = max_samples
        if enabled:
            os.makedirs(debug_dir, exist_ok=True)
            logger.info(f"调试输出已启用，输出目录: {os.path.abspath(debug_dir)}")
            logger.info(f"将保存最多 {max_samples} 个AKI和 {max_samples} 个非AKI样本的动态数据")
    
    def fit(self, train_dataset, test_dataset=None):
        """
        从数据集中提取特征和目标值，然后训练模型
        
        参数:
        - train_dataset: 训练数据集，包含静态特征、动态特征、目标值和持续时间
        - test_dataset: 测试数据集，可选
        
        返回:
        - train_loss: 训练损失
        """
        from torch.utils.data import DataLoader
        
        # 创建DataLoader
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        # 收集所有数据
        all_static_features = []
        all_dynamic_features = []
        all_targets = []
        all_durations = []  # 新增：收集durations
        
        print("收集训练数据...")
        for static_features, dynamic_features, targets, durations in train_loader:
            # 转换为NumPy数组
            if isinstance(static_features, torch.Tensor):
                static_features = static_features.numpy()
            if isinstance(dynamic_features, torch.Tensor):
                dynamic_features = dynamic_features.numpy()
            if isinstance(targets, torch.Tensor):
                targets = targets.numpy()
            if isinstance(durations, torch.Tensor):  # 新增：转换durations
                durations = durations.numpy()
            
            all_static_features.append(static_features)
            all_dynamic_features.append(dynamic_features)
            all_targets.append(targets)
            all_durations.append(durations)  # 新增：保存durations
        
        # 合并所有批次的数据
        static_features = np.vstack(all_static_features)
        dynamic_features = np.vstack([x.reshape(x.shape[0], -1, x.shape[-1]) for x in all_dynamic_features])
        targets = np.concatenate(all_targets)
        durations = np.concatenate(all_durations)  # 新增：合并durations
        
        print(f"收集完成，数据形状: static={static_features.shape}, dynamic={dynamic_features.shape}, targets={targets.shape}, durations={durations.shape}")
        
        # 首先处理特征，考虑AKI发生时间点
        flat_features = self.prepare_flat_features(static_features, dynamic_features, self.ml_max_steps, durations)
        
        # 调用子类的fit方法训练模型，传递已处理的特征
        self._fit(flat_features, targets)
        
        # 计算训练集上的损失，使用相同的特征
        train_preds = self.model.predict_proba(flat_features)[:, 1]
        train_loss = np.mean(-(targets * np.log(train_preds + 1e-10) + (1 - targets) * np.log(1 - train_preds + 1e-10)))
        print(f"训练集上的损失: {train_loss:.4f}")
        
        print(f"训练完成，训练集上的损失: {train_loss:.4f}")
        
        # 如果提供了测试集，计算测试集上的损失
        if test_dataset is not None:
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            
            all_static_features = []
            all_dynamic_features = []
            all_targets = []
            all_durations = []
            
            print("收集测试数据...")
            for static_features, dynamic_features, targets, durations in test_loader:
                # 转换为NumPy数组
                if isinstance(static_features, torch.Tensor):
                    static_features = static_features.numpy()
                if isinstance(dynamic_features, torch.Tensor):
                    dynamic_features = dynamic_features.numpy()
                if isinstance(targets, torch.Tensor):
                    targets = targets.numpy()
                if isinstance(durations, torch.Tensor):
                    durations = durations.numpy()
                
                all_static_features.append(static_features)
                all_dynamic_features.append(dynamic_features)
                all_targets.append(targets)
                all_durations.append(durations)
            
            # 合并所有批次的数据
            static_features = np.vstack(all_static_features)
            dynamic_features = np.vstack([x.reshape(x.shape[0], -1, x.shape[-1]) for x in all_dynamic_features])
            targets = np.concatenate(all_targets)
            durations = np.concatenate(all_durations)
            
            print(f"收集完成，测试数据形状: static={static_features.shape}, dynamic={dynamic_features.shape}, targets={targets.shape}, durations={durations.shape}")
            
            # 计算测试集上的损失，使用与训练集相同的特征处理逻辑
            flat_features = self.prepare_flat_features(static_features, dynamic_features, self.ml_max_steps, durations)
            test_preds = self.model.predict_proba(flat_features)[:, 1]
            test_loss = np.mean(-(targets * np.log(test_preds + 1e-10) + (1 - targets) * np.log(1 - test_preds + 1e-10)))
            
            print(f"测试集上的损失: {test_loss:.4f}")
        
        return train_loss
    def prepare_flat_features(self, static_features, dynamic_features, max_steps=None, durations=None):
        """
        将时序特征转换为定长特征，考虑AKI发生时间点
        
        参数:
        - static_features: 静态特征 [batch_size, static_dim]
        - dynamic_features: 动态特征 [batch_size, time_steps, dynamic_dim]
        - max_steps: 最大时间步数，如果为None则根据情况确定
        - durations: AKI发生的时间点 [batch_size]，如果为None则假设所有样本都没有AKI发生
        
        返回:
        - flat_features: 定长特征 [batch_size, static_dim + dynamic_dim * selected_steps]
        """
        # 检查输入类型，如果是PyTorch张量则转换为NumPy数组
        if isinstance(static_features, torch.Tensor):
            static_features = static_features.cpu().numpy()
        if isinstance(dynamic_features, torch.Tensor):
            dynamic_features = dynamic_features.cpu().numpy()
        if durations is not None and isinstance(durations, torch.Tensor):
            durations = durations.cpu().numpy()
            
        batch_size, time_steps, dynamic_dim = dynamic_features.shape
        
        # 获取h参数，如果未定义则使用默认值6
        h = getattr(self, 'h', 6)
        
        # 创建新的特征列表
        selected_features = []
        
        # 处理每个样本，与深度学习模型一致
        for i in range(batch_size):
            if durations is not None and durations[i] < 48:  # 如果该样本发生了AKI
                aki_time = int(durations[i])  # AKI发生的时间步
                
                # 如果AKI发生时间小于h，则跳过该样本
                if aki_time < h:
                    # 使用第一个时间步的数据
                    t = 0
                else:
                    # 使用t-h步前的特征预测
                    t = aki_time - h
                
                # 选择静态特征和动态特征的t时刻之前的数据
                selected_static = static_features[i:i+1]  # [1, static_dim]
                selected_dynamic = dynamic_features[i:i+1, :t+1]  # [1, t+1, dynamic_dim]
            else:  # 如果该样本未发生AKI
                # 随机选择(0,T-h)之间的时间步
                max_t = time_steps - h  # 确保有足够的后续时间步
                if max_t <= 0:
                    # 如果没有足够的时间步，使用第一个时间步
                    t = 0
                else:
                    t = np.random.randint(0, max_t)
                
                # 选择静态特征和动态特征的t时刻之前的数据
                selected_static = static_features[i:i+1]  # [1, static_dim]
                selected_dynamic = dynamic_features[i:i+1, :t+1]  # [1, t+1, dynamic_dim]
            
            # 将静态特征和动态特征合并并展平
            flat_static = selected_static.reshape(1, -1)  # [1, static_dim]
            
            # 始终使用固定长度的动态特征
            # 如果未指定max_steps，则使用一个默认值（例如10）
            if max_steps is None:
                max_steps = 10  # 默认使用10个时间步
                
            # 调整动态特征的时间步数
            current_steps = selected_dynamic.shape[1]
            if current_steps < max_steps:
                # 如果当前时间步数小于max_steps，需要填充
                padding = np.repeat(selected_dynamic[:, 0:1, :], max_steps - current_steps, axis=1)
                padded_dynamic = np.concatenate([padding,selected_dynamic], axis=1)
                flat_dynamic = padded_dynamic.reshape(1, -1)  # [1, max_steps * dynamic_dim]
            else:
                # 如果时间步数超过或等于max_steps，则截取
                flat_dynamic = selected_dynamic[:, :max_steps, :].reshape(1, -1)  # [1, max_steps * dynamic_dim]
            
            # 保存selected_dynamic为CSV文件以便查看（仅在调试模式下）
            if self._debug_enabled:
                # 检查是否还需要保存更多样本
                is_aki = (durations is not None and durations[i] > 0)
                
                if (is_aki and self._debug_aki_saved < self._debug_max_samples) or \
                   (not is_aki and self._debug_non_aki_saved < self._debug_max_samples):
                    
                    try:
                        # 确保目录存在
                        os.makedirs(self._debug_dir, exist_ok=True)
                        
                        # 将selected_dynamic转换为DataFrame
                        # selected_dynamic的形状为 [1, t+1, dynamic_dim]
                        batch_size, time_steps, feat_dim = selected_dynamic.shape
                        
                        # 展平为2D数组 [time_steps, feat_dim]
                        dynamic_data = selected_dynamic.reshape(-1, feat_dim)
                        
                        # 创建列名
                        columns = [f'dynamic_feat_{i+1}' for i in range(feat_dim)]
                        
                        # 创建DataFrame
                        df = pd.DataFrame(dynamic_data, columns=columns)
                        
                        # 添加时间步信息
                        df.insert(0, 'time_step', range(len(df)))
                        
                        # 添加样本信息
                        sample_type = 'aki' if is_aki else 'non_aki'
                        sample_count = self._debug_aki_saved if is_aki else self._debug_non_aki_saved
                        
                        # 保存为CSV
                        file_name = f'dynamic_data_{sample_type}_{sample_count + 1}.csv'
                        file_path = os.path.join(self._debug_dir, file_name)
                        df.to_csv(file_path, index=False)
                        
                        # 更新计数器
                        if is_aki:
                            self._debug_aki_saved += 1
                        else:
                            self._debug_non_aki_saved += 1
                            
                        logger.info(f"已保存{file_name}到 {os.path.abspath(file_path)}")
                        logger.info(f"已保存: {self._debug_aki_saved}个AKI样本, {self._debug_non_aki_saved}个非AKI样本")
                        
                    except Exception as e:
                        logger.error(f"保存selected_dynamic时出错: {e}")
            
            # 合并静态特征和动态特征
            flat_dynamic = np.zeros_like(flat_dynamic)
            combined_features = np.concatenate([flat_static, flat_dynamic], axis=1)
            selected_features.append(combined_features)
        
        # 将所有样本的特征堆叠成一个批次
        if selected_features:
            flat_features = np.vstack(selected_features)
            print(f"使用动态特征进行预测，特征维度={flat_features.shape}")
        else:
            # 如果没有有效样本，返回空数组
            flat_features = np.zeros((batch_size, static_features.shape[1]))
            print("警告: 没有有效样本，返回空数组")
        
        return flat_features
    
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
        if isinstance(static_features, torch.Tensor):
            static_np = static_features.cpu().numpy()
            device = static_features.device
        else:
            static_np = static_features
            device = torch.device('cpu')
            
        if isinstance(dynamic_features, torch.Tensor):
            dynamic_np = dynamic_features.cpu().numpy()
        else:
            dynamic_np = dynamic_features
        
        if isinstance(durations, torch.Tensor):
            durations_np = durations.cpu().numpy()
        else:
            durations_np = durations
        
        # 预测概率
        probs = self.predict_proba(static_np, dynamic_np, durations_np)
        
        # 转换为tensor
        aki_probs = torch.tensor(probs, device=device).unsqueeze(-1)
        
        # 计算损失
        loss = None
        if targets is not None:
            if isinstance(targets, np.ndarray):
                targets = torch.tensor(targets, device=device)
            
            # 使用二元交叉熵损失
            loss = F.binary_cross_entropy(aki_probs.squeeze(-1), targets.float())
        
        return aki_probs, loss


class RandomForestModel(BaseMLModel):
    """
    随机森林模型，用于AKI预测
    
    该模型将时序特征转换为定长特征，然后使用随机森林进行分类
    """
    def __init__(self, config):
        # 调用父类的__init__
        super().__init__()
        
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
        self.ml_max_steps = config.ml_max_steps
        
        logger.info(f"初始化随机森林模型: n_estimators={config.rf_n_estimators}, max_depth={config.rf_max_depth if config.rf_max_depth else '无限制'}")
    
    def _fit(self, flat_features, targets):
        """
        训练随机森林模型
        
        参数:
        - flat_features: 已处理的平展特征 [batch_size, feature_dim]
        - targets: 目标值，AKI是否发生 (0=未发生, 1=发生) [batch_size]
        
        返回:
        - self: 训练好的模型
        """
        # 生成特征名称 - 包括静态特征和动态特征
        feature_names = []
        
        # 静态特征名称 (static_1, static_2, ...)
        for i in range(self.static_dim):
            feature_names.append(f"static_{i+1}")
        
        # 动态特征名称 (dynamic_1_t1, dynamic_1_t2, ...)
        # 计算动态特征的时间步数
        remaining_features = flat_features.shape[1] - self.static_dim
        if remaining_features > 0 and self.dynamic_dim > 0:
            time_steps = remaining_features // self.dynamic_dim
            for t in range(time_steps):
                for d in range(self.dynamic_dim):
                    feature_names.append(f"dynamic_{d+1}_t{t+1}")
        
        # 保存特征名称
        self.feature_names = feature_names
        print(f"特征总数: {len(feature_names)}, 特征维度: {flat_features.shape[1]}")
        
        # 输出所有特征名称
        print("所有特征名称:\n静态特征:")
        for i in range(self.static_dim):
            print(f"{i+1}. {feature_names[i]}")
        
        if len(feature_names) > self.static_dim:
            print("\n动态特征 (前10个):")
            for i in range(self.static_dim, min(self.static_dim + 10, len(feature_names))):
                print(f"{i+1}. {feature_names[i]}")
            if len(feature_names) > self.static_dim + 10:
                print(f"... (共 {len(feature_names) - self.static_dim} 个动态特征)")        
        # 训练随机森林模型
        print(f"训练随机森林模型，特征维度: {flat_features.shape}")
        self.model.fit(flat_features, targets)
        
        # 输出特征重要性
        feature_importances = self.model.feature_importances_
        
        # 创建特征名称和重要性分数的字典
        feature_importance_dict = {}
        for i, name in enumerate(self.feature_names):
            if i < len(feature_importances):
                feature_importance_dict[name] = feature_importances[i]
        
        # 按重要性排序
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        # 输出所有特征的重要性分数
        print("特征重要性排名 (从高到低):")
        for i, (name, importance) in enumerate(sorted_features):
            print(f"{i+1}. {name}: {importance:.4f}")
        
        # 输出前10个重要特征
        top_indices = feature_importances.argsort()[-10:][::-1]
        print(f"随机森林模型训练完成，特征重要性前10索引: {top_indices}")
        
        # 输出前10个重要特征的名称和重要性分数
        print("前10个重要特征:")
        for i, idx in enumerate(top_indices):
            if idx < len(self.feature_names):
                print(f"{i+1}. {self.feature_names[idx]}: {feature_importances[idx]:.4f}")
            else:
                print(f"{i+1}. 特征索引{idx} (超出范围): {feature_importances[idx]:.4f}")
        
        return self
    
    def predict_proba(self, static_features, dynamic_features, durations=None):
        """
        预测AKI发生概率
        
        参数:
        - static_features: 静态特征 [batch_size, static_dim]
        - dynamic_features: 动态特征 [batch_size, time_steps, dynamic_dim]
        - durations: AKI发生的时间点 [batch_size]，如果为None则假设所有样本都没有AKI发生
        
        返回:
        - probs: AKI发生概率 [batch_size]
        """
        # 将时序特征转换为定长特征，考虑AKI发生时间点
        flat_features = self.prepare_flat_features(static_features, dynamic_features, self.ml_max_steps, durations)
        
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
        
        # 准备模型信息
        model_info = {
            'model': self.model,
            'feature_version': 'v2',  # 新版特征处理逻辑
            'h': self.h,
            'ml_max_steps': self.ml_max_steps,
            'static_dim': self.static_dim,
            'dynamic_dim': self.dynamic_dim
        }
        
        # 保存模型及其信息
        joblib.dump(model_info, path)
        logger.info(f"随机森林模型已保存到 {path} (特征版本: v2)")
    
    def load(self, path):
        """
        加载模型
        
        参数:
        - path: 模型加载路径
        
        返回:
        - self: 加载好的模型
        """
        # 加载模型
        try:
            # 尝试加载新格式的模型（包含特征处理信息）
            model_info = joblib.load(path)
            
            if isinstance(model_info, dict) and 'model' in model_info:
                # 新格式模型
                self.model = model_info['model']
                
                # 加载特征处理参数
                if 'feature_version' in model_info:
                    feature_version = model_info['feature_version']
                    self._legacy_mode = (feature_version == 'v1')
                    
                    if self._legacy_mode:
                        logger.info(f"加载旧版模型（特征版本: {feature_version}），将使用旧版特征处理逻辑")
                    else:
                        logger.info(f"加载新版模型（特征版本: {feature_version}）")
                else:
                    # 如果没有版本信息，则默认为旧版
                    self._legacy_mode = True
                    logger.info("模型没有版本信息，将使用旧版特征处理逻辑")
                
                # 加载其他参数
                if 'h' in model_info:
                    self.h = model_info['h']
                if 'ml_max_steps' in model_info:
                    self.ml_max_steps = model_info['ml_max_steps']
                if 'static_dim' in model_info:
                    self.static_dim = model_info['static_dim']
                if 'dynamic_dim' in model_info:
                    self.dynamic_dim = model_info['dynamic_dim']
                
                logger.info(f"随机森林模型已从 {path} 加载，h={self.h}, ml_max_steps={self.ml_max_steps}")
            else:
                # 旧格式模型（直接保存的模型对象）
                self.model = model_info
                self._legacy_mode = True
                logger.info(f"随机森林模型已从 {path} 加载（旧格式），将使用旧版特征处理逻辑")
        except Exception as e:
            logger.error(f"加载模型时出错: {e}")
            raise
        
        return self


class LogisticRegressionModel(BaseMLModel):
    """
    逻辑回归模型，用于AKI预测
    
    该模型将时序特征转换为定长特征，然后使用逻辑回归进行分类
    """
    def __init__(self, config):
        # 调用父类的__init__
        super().__init__()
        
        self.config = config
        
        # 初始化逻辑回归分类器
        self.model = LogisticRegression(
            C=config.lr_C,
            penalty=config.lr_penalty,
            solver=config.lr_solver,
            max_iter=config.lr_max_iter,
            random_state=42
        )
        
        # 保存模型配置
        self.static_dim = config.static_dim
        self.dynamic_dim = config.dynamic_dim
        self.h = config.h
        self.ml_max_steps = config.ml_max_steps
        
        logger.info(f"初始化逻辑回归模型: C={config.lr_C}, penalty={config.lr_penalty}, solver={config.lr_solver}")
    
    def _fit(self, flat_features, targets):
        """
        训练逻辑回归模型
        
        参数:
        - flat_features: 已处理的平展特征 [batch_size, feature_dim]
        - targets: 目标值，AKI是否发生 (0=未发生, 1=发生) [batch_size]
        
        返回:
        - self: 训练好的模型
        """
        # 训练逻辑回归模型
        logger.info(f"训练逻辑回归模型，特征维度: {flat_features.shape}")
        self.model.fit(flat_features, targets)
        
        # 输出模型系数
        if hasattr(self.model, 'coef_'):
            coef = self.model.coef_[0]
            logger.info(f"逻辑回归模型训练完成，系数前10: {coef.argsort()[-10:][::-1]}")
        
        return self
    
    def predict_proba(self, static_features, dynamic_features, durations=None):
        """
        预测AKI发生概率
        
        参数:
        - static_features: 静态特征 [batch_size, static_dim]
        - dynamic_features: 动态特征 [batch_size, time_steps, dynamic_dim]
        - durations: AKI发生的时间点 [batch_size]，如果为None则假设所有样本都没有AKI发生
        
        返回:
        - probs: AKI发生概率 [batch_size]
        """
        # 将时序特征转换为定长特征，考虑AKI发生时间点
        flat_features = self.prepare_flat_features(static_features, dynamic_features, self.ml_max_steps, durations)
        
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
        
        # 准备模型信息
        model_info = {
            'model': self.model,
            'feature_version': 'v2',  # 新版特征处理逻辑
            'h': self.h,
            'ml_max_steps': self.ml_max_steps,
            'static_dim': self.static_dim,
            'dynamic_dim': self.dynamic_dim
        }
        
        # 保存模型及其信息
        joblib.dump(model_info, path)
        logger.info(f"逻辑回归模型已保存到 {path} (特征版本: v2)")
    
    def load(self, path):
        """
        加载模型
        
        参数:
        - path: 模型加载路径
        
        返回:
        - self: 加载好的模型
        """
        # 加载模型
        try:
            # 尝试加载新格式的模型（包含特征处理信息）
            model_info = joblib.load(path)
            
            if isinstance(model_info, dict) and 'model' in model_info:
                # 新格式模型
                self.model = model_info['model']
                
                # 加载特征处理参数
                if 'feature_version' in model_info:
                    feature_version = model_info['feature_version']
                    self._legacy_mode = (feature_version == 'v1')
                    
                    if self._legacy_mode:
                        logger.info(f"加载旧版模型（特征版本: {feature_version}），将使用旧版特征处理逻辑")
                    else:
                        logger.info(f"加载新版模型（特征版本: {feature_version}）")
                else:
                    # 如果没有版本信息，则默认为旧版
                    self._legacy_mode = True
                    logger.info("模型没有版本信息，将使用旧版特征处理逻辑")
                
                # 加载其他参数
                if 'h' in model_info:
                    self.h = model_info['h']
                if 'ml_max_steps' in model_info:
                    self.ml_max_steps = model_info['ml_max_steps']
                if 'static_dim' in model_info:
                    self.static_dim = model_info['static_dim']
                if 'dynamic_dim' in model_info:
                    self.dynamic_dim = model_info['dynamic_dim']
                
                logger.info(f"逻辑回归模型已从 {path} 加载，h={self.h}, ml_max_steps={self.ml_max_steps}")
            else:
                # 旧格式模型（直接保存的模型对象）
                self.model = model_info
                self._legacy_mode = True
                logger.info(f"逻辑回归模型已从 {path} 加载（旧格式），将使用旧版特征处理逻辑")
        except Exception as e:
            logger.error(f"加载模型时出错: {e}")
            raise
        
        return self

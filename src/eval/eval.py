import sys
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, classification_report, precision_recall_curve, average_precision_score
from lifelines.utils import concordance_index
from imblearn.under_sampling import RandomUnderSampler

# 添加项目路径到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.model.model import AKIConfig, AKIPredictor, prepare_data
from src.model.ml_model import MLConfig, RandomForestModel, LogisticRegressionModel

logger = logging.getLogger(__name__)

def seed_everything(seed=3407):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_model(model_path, config=None):
    """
    加载训练好的模型
    
    参数:
    - model_path: 模型文件路径
    - config: 模型配置，如果为None则使用默认配置
    
    返回:
    - 加载好的模型
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在")
    
    # 检查文件扩展名，判断是机器学习模型还是PyTorch模型
    if model_path.endswith('.pkl'):
        # 机器学习模型，使用joblib加载
        import joblib
        
        # 根据配置创建相应的模型
        if config.model_type == 'RandomForest':
            from src.model.ml_model import RandomForestModel
            model = RandomForestModel(config)
        elif config.model_type == 'LogisticRegression':
            from src.model.ml_model import LogisticRegressionModel
            model = LogisticRegressionModel(config)
        else:
            raise ValueError(f"不支持的机器学习模型类型: {config.model_type}")
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件 {model_path} 不存在，请先训练并保存模型")
            
        try:
            # 加载模型
            model.load(model_path)
            print(f"成功加载{config.model_type}模型: {model_path}")
        except Exception as e:
            print(f"加载模型失败: {e}")
            print("请确保模型已经被训练并正确保存，且文件格式为.pkl")
            raise
        
        return model
    else:
        # PyTorch模型，使用现有的加载方式
        try:
            # 尝试直接加载完整模型
            model = torch.load(model_path, weights_only = False)
            print("成功加载完整模型")
        except Exception as e:
            print(f"无法加载完整模型: {e}")
            print("尝试加载模型权重...")
            
            # 如果直接加载失败，创建新模型并加载权重
            if config is None:
                config = AKIConfig()
            
            model = AKIPredictor(config)
            model.load_state_dict(torch.load(model_path))
            print("成功加载模型权重")
        
        model.eval()  # 设置为评估模式
        return model

def plot_confusion_matrix(y_true, y_pred, output_dir='data/results', threshold=0.5):
    """
    生成并可视化混淆矩阵
    
    参数:
    - y_true: 真实标签
    - y_pred: 预测风险评分
    - output_dir: 输出目录
    - threshold: 分类阈值，默认为0.5
    
    返回:
    - cm: 混淆矩阵
    - report: 分类报告
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 将风险评分转换为二分类预测（大于阈值的视为预测发生AKI）
    y_pred_binary = (y_pred > threshold).astype(int)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred_binary)
    
    # 生成分类报告
    report = classification_report(y_true, y_pred_binary, target_names=['no AKI', 'AKI'], output_dict=True)
    
    # 打印分类报告
    print("分类报告:")
    print(classification_report(y_true, y_pred_binary, target_names=['未发生AKI', '发生AKI']))
    
    # 检查是否总是预测同一类别
    if np.all(y_pred_binary == 0):
        print("警告：模型总是预测'未发生AKI'，可能存在严重偏差！")
    elif np.all(y_pred_binary == 1):
        print("警告：模型总是预测'发生AKI'，可能存在严重偏差！")
    
    # 可视化混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted no AKI', 'Predicted AKI'],
                yticklabels=['Actual no AKI', 'Actual AKI'])
    plt.ylabel('Actual labels')
    plt.xlabel('Predicted labels')
    plt.title('Confusion Matrix')
    
    # 保存图像
    confusion_matrix_file = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_file)
    plt.close()
    print(f"混淆矩阵已保存到 {confusion_matrix_file}")
    
    # 计算ROC曲线和AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    # 可视化ROC曲线
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    
    # 保存ROC曲线
    roc_curve_file = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(roc_curve_file)
    plt.close()
    print(f"ROC曲线已保存到 {roc_curve_file}")
    
    # 绘制风险评分分布
    plt.figure(figsize=(10, 6))
    
    # 分别绘制两类样本的风险评分分布
    plt.hist(y_pred[y_true==0], bins=20, alpha=0.5, label='no AKI')
    plt.hist(y_pred[y_true==1], bins=20, alpha=0.5, label='AKI')
    
    plt.xlabel('风险评分')
    plt.ylabel('样本数量')
    plt.title('不同类别的风险评分分布')
    plt.legend()
    
    # 保存风险评分分布图
    risk_dist_file = os.path.join(output_dir, 'risk_score_distribution.png')
    plt.savefig(risk_dist_file)
    plt.close()
    print(f"风险评分分布图已保存到 {risk_dist_file}")
    
    return cm, report

def prepare_data_with_random_cutoff(test_data, time_steps=48, h=6):
    """
    准备评估数据，为每个样本随机选择一个截止时间点
    
    参数:
    - test_data: 测试数据DataFrame
    - time_steps: 时间步数
    - h: 提前预测步数
    
    返回:
    - static_features: 静态特征 [batch_size, static_dim]
    - dynamic_features: 动态特征 [batch_size, time_steps, dynamic_dim]
    - targets: 目标值，在随机截止时间内是否会发生AKI (0=未发生, 1=发生) [batch_size]
    - durations: AKI发生的时间点 [batch_size]
    - cutoff_times: 随机选择的截止时间点 [batch_size]
    """
    # 首先使用原始prepare_data获取基本数据
    static_features, dynamic_features, original_targets, original_durations = prepare_data(test_data, time_steps, h=0)  # 设置h=0以获取所有数据
    
    batch_size = static_features.shape[0]
    
    # 为每个样本随机选择一个截止时间点
    cutoff_times = []
    new_targets = []
    
    for i in range(batch_size):
        # 获取当前样本的AKI发生时间
        aki_time = original_durations[i].item()
        
        # 确定可选的截止时间范围 (0, min(time_steps-h, aki_time-h) if aki_time != time_steps else time_steps-h)
        if aki_time == time_steps:  # 未发生AKI
            max_cutoff = time_steps - h
        else:  # 发生了AKI
            max_cutoff = min(time_steps - h, aki_time - h)
        
        # 确保max_cutoff至少为1
        max_cutoff = max(1, max_cutoff)
        
        # 随机选择一个截止时间点 (1到max_cutoff之间)
        cutoff_time = np.random.randint(1, max_cutoff + 1)
        cutoff_times.append(cutoff_time)
        
        # 确定在截止时间内是否会发生AKI
        # 只有当AKI发生时间 <= 截止时间+h 且 AKI发生时间 != time_steps (未发生) 时，才认为会发生AKI
        will_aki_occur = (aki_time <= cutoff_time + h) and (aki_time != time_steps)
        new_targets.append(1 if will_aki_occur else 0)
    
    # 转换为张量
    cutoff_times = torch.tensor(cutoff_times, dtype=torch.float32)
    new_targets = torch.tensor(new_targets, dtype=torch.float32)
    
    # 打印统计信息
    print(f"随机截止时间点范围: {cutoff_times.min().item()} - {cutoff_times.max().item()}")
    print(f"原始目标值: 正样本数量: {torch.sum(original_targets == 1).item()}, 负样本数量: {torch.sum(original_targets == 0).item()}")
    print(f"新目标值: 正样本数量: {torch.sum(new_targets == 1).item()}, 负样本数量: {torch.sum(new_targets == 0).item()}")
    
    return static_features, dynamic_features, new_targets, original_durations, cutoff_times

def evaluate_model(model, test_data, time_steps=48, h=6, output_dir='data/results', threshold=0.5):
    """
    评估模型性能
    
    参数:
    - model: 训练好的模型
    - test_data: 测试数据DataFrame
    - time_steps: 时间步数
    - h: 提前预测步数，小于这个时间步发生的数据先筛除
    - output_dir: 输出目录
    - threshold: 分类阈值，默认为0.5
    
    返回:
    - accuracy: 准确率
    - results_df: 包含预测概率、持续时间和事件的DataFrame
    - metrics: 包含准确率、精确度、召回率和F1分数的字典
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备数据
    try:
        print("准备数据...")
        print(f"提前预测步数 h = {h}")
        
        # 使用新的数据准备函数，为每个样本随机选择一个截止时间点
        static_features, dynamic_features, targets, durations, cutoff_times = prepare_data_with_random_cutoff(test_data, time_steps, h=h)
        print(f"数据准备完成: {static_features.shape[0]} 个样本")
        
        # 计算正负样本数量
        positive_count = torch.sum(targets == 1).item()
        negative_count = torch.sum(targets == 0).item()
        print(f"原始数据: 正样本数量: {positive_count}, 负样本数量: {negative_count}, 正负比例: {positive_count/negative_count if negative_count > 0 else 'inf':.4f}")
        
        # 执行欠采样以平衡类别
        if positive_count < negative_count:
            print("执行欠采样以平衡类别...")
            # 准备数据用于欠采样
            indices = np.arange(len(targets))
            X = np.column_stack([indices])
            y = targets.cpu().numpy()
            
            # 创建欠采样器
            undersampler = RandomUnderSampler(sampling_strategy='majority', random_state=42)
            X_resampled, y_resampled = undersampler.fit_resample(X, y)
            
            # 获取欠采样后的索引
            resampled_indices = X_resampled[:, 0].astype(int)
            
            # 应用欠采样
            static_features = static_features[resampled_indices]
            dynamic_features = dynamic_features[resampled_indices]
            targets = targets[resampled_indices]
            durations = durations[resampled_indices]
            cutoff_times = cutoff_times[resampled_indices]
            
            # 计算欠采样后的正负样本数量
            positive_count = torch.sum(targets == 1).item()
            negative_count = torch.sum(targets == 0).item()
            print(f"欠采样后: 正样本数量: {positive_count}, 负样本数量: {negative_count}, 正负比例: {positive_count/negative_count if negative_count > 0 else 'inf':.4f}")
            print(f"欠采样后样本总数: {len(targets)}")
    except Exception as e:
        print(f"数据准备失败: {e}")
        raise
    
    # 检查NaN/Inf值
    def check_nan_inf(tensor, name):
        if torch.isnan(tensor).any():
            print(f"警告: {name} 中存在NaN值")
            nan_indices = torch.where(torch.isnan(tensor))[0]
            print(f"NaN索引: {nan_indices[:10]}{'...' if len(nan_indices) > 10 else ''}")
        if torch.isinf(tensor).any():
            print(f"警告: {name} 中存在Inf值")
            inf_indices = torch.where(torch.isinf(tensor))[0]
            print(f"Inf索引: {inf_indices[:10]}{'...' if len(inf_indices) > 10 else ''}")
    
    check_nan_inf(static_features, "静态特征")
    check_nan_inf(dynamic_features, "动态特征")
    check_nan_inf(targets, "目标")
    check_nan_inf(durations, "持续时间")
    
    # 检查模型类型，获取设备
    is_pytorch_model = hasattr(model, 'parameters')
    
    if is_pytorch_model:
        # PyTorch模型，获取设备并移动数据
        device = next(model.parameters()).device
        print(f"模型所在设备: {device}")
        
        # 将所有张量移动到相同的设备上
        static_features = static_features.to(device)
        dynamic_features = dynamic_features.to(device)
        targets = targets.to(device)
        durations = durations.to(device)
    else:
        # 机器学习模型，使用CPU
        print("机器学习模型，使用CPU处理")
    
    # 获取AKI发生概率
    if is_pytorch_model:
        # PyTorch模型使用forward方法
        with torch.no_grad():
            try:
                # 设置为评估模式（非训练模式）
                # 对于每个样本，只使用截止时间点之前的动态特征
                batch_size = static_features.shape[0]
                all_probs = []
                
                for i in range(batch_size):
                    # 获取当前样本的截止时间点
                    cutoff = int(cutoff_times[i].item())
                    
                    # 提取截止时间点之前的动态特征
                    sample_dynamic = dynamic_features[i:i+1, :cutoff, :]
                    sample_static = static_features[i:i+1, :]
                    
                    # 使用模型进行预测
                    sample_probs, _ = model(sample_static, sample_dynamic, is_training=False)
                    all_probs.append(sample_probs)
                
                # 合并所有预测结果
                aki_probs = torch.cat(all_probs, dim=0)
                
                # 检查输出是否包含NaN/Inf
                check_nan_inf(aki_probs, "AKI发生概率")
            except Exception as e:
                print(f"模型推理失败: {e}")
                raise
    else:
        # 机器学习模型使用predict_proba方法
        try:
            # 转换为NumPy数组
            static_np = static_features.numpy() if isinstance(static_features, torch.Tensor) else static_features
            dynamic_np = dynamic_features.numpy() if isinstance(dynamic_features, torch.Tensor) else dynamic_features
            cutoff_times_np = cutoff_times.numpy() if isinstance(cutoff_times, torch.Tensor) else cutoff_times
            
            # 对于每个样本，只使用截止时间点之前的动态特征
            batch_size = static_np.shape[0]
            all_probs = []
            
            for i in range(batch_size):
                # 获取当前样本的截止时间点
                cutoff = int(cutoff_times_np[i])
                
                # 提取截止时间点之前的动态特征
                sample_dynamic = dynamic_np[i:i+1, :cutoff, :]
                sample_static = static_np[i:i+1, :]
                
                # 使用模型进行预测
                sample_probs = model.predict_proba(sample_static, sample_dynamic)
                all_probs.append(sample_probs)
            
            # 合并所有预测结果
            probs = np.array(all_probs)
            
            # 转换为PyTorch张量以保持与现有代码的兼容性
            aki_probs = torch.tensor(probs).unsqueeze(-1)
            
            print(f"机器学习模型预测概率范围: {probs.min():.4f} - {probs.max():.4f}")
        except Exception as e:
            print(f"机器学习模型推理失败: {e}")
            raise
    
    # 打印形状信息
    print(f"aki_probs shape: {aki_probs.shape}")
    print(f"targets shape: {targets.shape}")
    print(f"durations shape: {durations.shape}")
    
    # 将概率转换为二分类预测
    aki_preds = (aki_probs.squeeze(-1) > threshold).float()
    
    # 转换为NumPy数组进行评估
    if is_pytorch_model:
        # PyTorch模型需要将张量移动到CPU并转换为NumPy
        targets_np = targets.cpu().numpy()
        aki_preds_np = aki_preds.cpu().numpy()
        aki_probs_np = aki_probs.squeeze(-1).cpu().numpy()
    else:
        # 机器学习模型的输出可能已经是NumPy数组
        targets_np = targets.numpy() if isinstance(targets, torch.Tensor) else targets
        aki_preds_np = aki_preds.numpy() if isinstance(aki_preds, torch.Tensor) else aki_preds
        aki_probs_np = aki_probs.squeeze(-1).numpy() if isinstance(aki_probs, torch.Tensor) else aki_probs.squeeze(-1)
    
    # 计算分类指标
    accuracy = accuracy_score(targets_np, aki_preds_np)
    precision = precision_score(targets_np, aki_preds_np, zero_division=0)
    recall = recall_score(targets_np, aki_preds_np, zero_division=0)
    f1 = f1_score(targets_np, aki_preds_np, zero_division=0)
    
    print(f"准确率: {accuracy:.4f}")
    print(f"精确度: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    # 生成混淆矩阵
    cm, report = plot_confusion_matrix(
        targets_np, 
        aki_probs_np, 
        output_dir=output_dir,
        threshold=threshold
    )
    
    # 注意：前面已经转换了aki_probs_np和targets_np
    # 这里只需要转换durations_np（如果还没有转换）
    if is_pytorch_model:
        durations_np = durations.cpu().numpy()
    else:
        durations_np = durations.numpy() if isinstance(durations, torch.Tensor) else durations
    
    # 打印形状信息
    print(f"aki_probs shape: {aki_probs_np.shape}")
    print(f"targets shape: {targets_np.shape}")
    print(f"durations shape: {durations_np.shape}")
    
    # 将概率转换为二分类预测
    binary_pred = (aki_probs_np > threshold).astype(int)
    
    # 计算AUC
    fpr, tpr, _ = roc_curve(targets_np, aki_probs_np)
    roc_auc = auc(fpr, tpr)
    print(f"AUC: {roc_auc:.4f}")
    
    # 创建结果数据帧
    results_df = pd.DataFrame({
        'aki_prob': aki_probs_np,
        'aki_pred': binary_pred,
        'actual': targets_np,
        'duration': durations_np,
        'cutoff_time': cutoff_times.cpu().numpy() if isinstance(cutoff_times, torch.Tensor) else cutoff_times
    })
    
    # 保存结果数据帧
    results_file = os.path.join(output_dir, 'results.csv')
    results_df.to_csv(results_file, index=False)
    print(f"结果已保存到 {results_file}")
    
    # 绘制ROC曲线
    # 注意：前面已经计算了fpr和tpr，这里直接使用
    # fpr, tpr, _ = roc_curve(targets_np, aki_probs_np)  # 这行已在前面计算
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    
    # 保存ROC曲线
    roc_file = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(roc_file)
    plt.close()
    print(f"ROC曲线已保存到 {roc_file}")
    
    # 收集指标
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': roc_auc
    }
    
    return accuracy, results_df, metrics

def evaluate_prediction_horizons(risk_probs, target_matrix, targets, durations, horizons, threshold=0.5, output_dir='data/results'):
    """
    评估不同预测提前时间的性能
    
    参数:
    - risk_probs: 风险概率矩阵 [batch_size, time_steps, prediction_horizon]
    - target_matrix: 目标矩阵 [batch_size, time_steps, prediction_horizon]
    - targets: 目标标签 [batch_size]
    - durations: 持续时间 [batch_size]
    - horizons: 要评估的预测提前时间列表，例如[6, 8, 10, 12]
    - threshold: 分类阈值
    - output_dir: 输出目录
    
    返回:
    - horizon_metrics: 不同预测提前时间的评估指标字典
    """
    print("\n评估不同预测提前时间的性能...")
    
    batch_size, time_steps, prediction_horizon = risk_probs.shape
    horizon_metrics = {}
    
    # 创建用于存储所有指标的DataFrame
    all_metrics_data = []
    
    # 对每个预测提前时间进行评估
    for horizon in horizons:
        if horizon >= prediction_horizon:
            print(f"警告: 预测提前时间 {horizon} 超出了模型的预测范围 {prediction_horizon}，将跳过")
            continue
            
        print(f"评估预测提前时间: {horizon} 个时间步")
        
        # 提取特定预测提前时间的风险概率和目标
        horizon_risk_probs = risk_probs[:, :, horizon-1]  # 索引从0开始，所以需要-1
        horizon_targets = target_matrix[:, :, horizon-1]
        
        # 对每个样本取最大风险概率
        sample_risk_probs = horizon_risk_probs.max(axis=1)
        sample_targets = horizon_targets.max(axis=1)
        
        # 将风险概率转换为二分类预测
        binary_pred = (sample_risk_probs > threshold).astype(int)
        
        # 计算各种评估指标
        # 注意: 如果样本中没有正例或负例，部分指标可能无法计算
        try:
            tn, fp, fn, tp = confusion_matrix(sample_targets, binary_pred).ravel()
            accuracy = accuracy_score(sample_targets, binary_pred)
            precision = precision_score(sample_targets, binary_pred, zero_division=0)
            recall = recall_score(sample_targets, binary_pred, zero_division=0)
            f1 = f1_score(sample_targets, binary_pred, zero_division=0)
            
            # 计算ROC曲线和AUC
            fpr, tpr, _ = roc_curve(sample_targets, sample_risk_probs)
            roc_auc = auc(fpr, tpr)
        except Exception as e:
            print(f"计算指标时出错: {e}")
            tn, fp, fn, tp = 0, 0, 0, 0
            accuracy = precision = recall = f1 = roc_auc = 0
        
        # 打印指标
        print(f"  混淆矩阵: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        print(f"  准确率: {accuracy:.4f}, 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1: {f1:.4f}, AUC: {roc_auc:.4f}")
        
        # 绘制ROC曲线
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for Prediction Horizon {horizon}')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, f'roc_curve_horizon_{horizon}.png'))
        plt.close()
        
        # 存储指标
        horizon_metrics[horizon] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': roc_auc,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp
        }
        
        # 添加到指标数据中
        all_metrics_data.append({
            'horizon': horizon,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': roc_auc
        })
    
    # 创建指标DataFrame并保存
    if all_metrics_data:
        metrics_df = pd.DataFrame(all_metrics_data)
        metrics_df.to_csv(os.path.join(output_dir, 'horizon_metrics.csv'), index=False)
        
        # 绘制不同预测提前时间的性能对比图
        plt.figure(figsize=(12, 8))
        
        # 绘制不同指标
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        markers = ['o', 's', '^', 'D', '*']
        
        for i, metric in enumerate(metrics):
            plt.plot(metrics_df['horizon'], metrics_df[metric], 
                     label=metric, color=colors[i], marker=markers[i], linewidth=2)
        
        plt.xlabel('预测提前时间 (时间步)')
        plt.ylabel('指标值')
        plt.title('不同预测提前时间的性能对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'horizon_performance_comparison.png'))
        plt.close()
        
        print(f"不同预测提前时间的性能指标已保存到 {os.path.join(output_dir, 'horizon_metrics.csv')}")
    
    return horizon_metrics

def plot_survival_curves(results_df, num_groups=3, output_dir='data/results'):
    """
    根据风险评分绘制生存曲线
    
    参数:
    - results_df: 包含风险评分、持续时间和事件的DataFrame
    - num_groups: 风险组数量
    - output_dir: 输出目录
    """
    # 根据风险评分将患者分为高、中、低风险组
    results_df['risk_group'] = pd.qcut(results_df['risk_score'], num_groups, labels=False)
    
    plt.figure(figsize=(10, 6))
    colors = ['green', 'blue', 'red']
    labels = ['low', 'mid', 'high']
    
    for i in range(num_groups):
        group_df = results_df[results_df['risk_group'] == i]
        
        # 计算生存率
        sorted_times = sorted(set(group_df['duration']))
        survival_rates = []
        
        for t in sorted_times:
            # 在时间t之后仍然存活的比例
            survival_rate = (group_df['actual_time'] > t).mean()
            survival_rates.append(survival_rate)
        
        plt.step(sorted_times, survival_rates, where='post', color=colors[i], label=labels[i])
    
    plt.xlabel('timestep')
    plt.ylabel('survival probility')
    plt.title('survival curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图像
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'survival_curves.png')
    plt.savefig(output_file)
    plt.close()
    
    print(f"生存曲线已保存到 {output_file}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='评估AKI预测模型')
    parser.add_argument('--model', type=str, default='data/ckpt/trained-model-.pt', help='模型路径')
    parser.add_argument('--data', type=str, default='data/aki_hypertension_data_processed_eval.csv', help='测试数据路径')
    parser.add_argument('--time_steps', type=int, default=48, help='时间步数')
    parser.add_argument('--h', type=int, default=6, help='提前预测步数，小于这个时间步发生的数据先筛除')
    parser.add_argument('--model_type', type=str, default='RWKV', choices=['RWKV', 'LSTM', 'GRU', 'Transformer', 'RandomForest', 'LogisticRegression'], help='模型类型: RWKV, LSTM, GRU, Transformer, RandomForest, LogisticRegression')
    
    # LSTM特定参数
    parser.add_argument('--lstm_layers', type=int, default=1, help='LSTM层数')
    parser.add_argument('--lstm_bidirectional', action='store_true', help='是否使用双向LSTM')
    
    # GRU特定参数
    parser.add_argument('--gru_layers', type=int, default=1, help='GRU层数')
    parser.add_argument('--gru_bidirectional', action='store_true', help='是否使用双向GRU')
    
    # Transformer特定参数
    parser.add_argument('--attn_dropout', type=float, default=0.1, help='Transformer注意力机制的dropout率')
    parser.add_argument('--ff_activation', type=str, default='gelu', choices=['gelu', 'relu', 'silu', 'mish'], help='Transformer前馈网络的激活函数类型')
    
    # 随机森林特定参数
    parser.add_argument('--rf_n_estimators', type=int, default=100, help='随机森林中树的数量')
    parser.add_argument('--rf_max_depth', type=int, default=None, help='随机森林中树的最大深度，None表示无限制')
    parser.add_argument('--rf_min_samples_split', type=int, default=2, help='分裂内部节点所需的最小样本数')
    parser.add_argument('--rf_min_samples_leaf', type=int, default=1, help='叶节点所需的最小样本数')
    
    # 逻辑回归特定参数
    parser.add_argument('--lr_C', type=float, default=1.0, help='逻辑回归正则化强度的倒数，较小的值表示更强的正则化')
    parser.add_argument('--lr_penalty', type=str, default='l2', choices=['l1', 'l2', 'elasticnet', 'none'], help='逻辑回归使用的惩罚类型')
    parser.add_argument('--lr_solver', type=str, default='lbfgs', choices=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], help='逻辑回归优化问题的算法')
    parser.add_argument('--lr_max_iter', type=int, default=100, help='逻辑回归求解器收敛的最大迭代次数')
    
    # 机器学习模型共用参数
    parser.add_argument('--ml_max_steps', type=int, default=48, help='机器学习模型使用的最大时间步数')
    parser.add_argument('--output', type=str, default='data/results', help='输出结果目录')
    parser.add_argument('--groups', type=int, default=3, help='风险组数量')
    parser.add_argument('--threshold', type=float, default=0.515, help='分类阈值，默认为0.5，用于生成混淆矩阵')
    parser.add_argument('--weighted_loss', action='store_true', help='是否使用加权损失函数处理数据不平衡')
    parser.add_argument('--pos_weight', type=float, default=7.0, help='正样本权重，默认为7.0（多数类与少数类的比例）')
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output, exist_ok=True)
    
    # 加载测试数据
    print(f'加载测试数据: {args.data}')
    test_data_path = args.data
    
    # 加载测试数据
    test_data = pd.read_csv(test_data_path)
    print(f"测试数据大小: {test_data.shape[0]} 行 × {test_data.shape[1]} 列")
    
    # 检查数据中的缺失值
    missing_values = test_data.isnull().sum().sum()
    if missing_values > 0:
        print(f"警告: 测试数据中存在 {missing_values} 个缺失值")
        print("各列缺失值数量:")
        missing_cols = test_data.isnull().sum()
        print(missing_cols[missing_cols > 0])
    
    # 准备数据以获取特征维度
    print("准备数据以获取特征维度...")
    try:
        # 准备数据
        print(f"提前预测步数 h = {args.h}")
        static_features, dynamic_features, targets, durations = prepare_data(test_data, args.time_steps, h=args.h)
        
        # 获取特征维度
        static_dim = static_features.shape[1]  # 静态特征维度
        dynamic_dim = dynamic_features.shape[2]  # 动态特征维度
        time_steps = dynamic_features.shape[1]  # 时间步数
        
        print(f"特征维度: static_dim={static_dim}, dynamic_dim={dynamic_dim}, time_steps={time_steps}")
        
        # 根据模型类型选择不同的配置和模型类
        if args.model_type in ['RandomForest', 'LogisticRegression']:
            # 使用机器学习模型配置
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
            
            # 检查模型文件
            if not args.model:
                raise ValueError("请指定模型文件路径。机器学习模型需要先训练并保存，然后才能进行评估。")
            
            if not os.path.exists(args.model):
                raise FileNotFoundError(f"模型文件 {args.model} 不存在。请先训练并保存模型，然后再进行评估。")
            
            # 选择正确的模型类
            if args.model_type == 'RandomForest':
                model = RandomForestModel(config)
                model.load(args.model)
                print(f"成功加载随机森林模型: {args.model}")
                print(f"使用随机森林模型，树数量: {args.rf_n_estimators}, 最大深度: {args.rf_max_depth if args.rf_max_depth else '无限制'}")
            else:  # LogisticRegression
                model = LogisticRegressionModel(config)
                model.load(args.model)
                print(f"成功加载逻辑回归模型: {args.model}")
                print(f"使用逻辑回归模型，正则化强度: {args.lr_C}, 正则化类型: {args.lr_penalty}")
                print(f"使用逻辑回归模型，正则化强度: {args.lr_C}, 正则化类型: {args.lr_penalty}")
        else:
            # 使用深度学习模型配置
            config = AKIConfig(
                static_dim=static_dim,
                dynamic_dim=dynamic_dim,
                ctx_len=time_steps,
                embed_dim=128,
                n_layer=3,
                n_head=4,
                h=args.h,  # 提前预测步数
                model_type=args.model_type,  # 模型类型
                
                # LSTM特定参数
                lstm_layers=args.lstm_layers,
                lstm_bidirectional=args.lstm_bidirectional,
                
                # GRU特定参数
                gru_layers=args.gru_layers,
                gru_bidirectional=args.gru_bidirectional,
                
                # Transformer特定参数
                attn_dropout=args.attn_dropout,
                ff_activation=args.ff_activation
            )
            
            print(f"使用模型类型: {args.model_type}")
            
            if args.weighted_loss:
                config.pos_weight = args.pos_weight
                print(f"使用加权损失函数，正样本权重: {args.pos_weight}")
            
            # 加载模型
            model = load_model(args.model, config)
            print(f"成功加载模型: {args.model}")
        
    except Exception as e:
        print(f"初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 评估模型
    accuracy, results_df, metrics = evaluate_model(model, test_data, args.time_steps, args.h, args.output, args.threshold)
    
    # 保存评估结果
    result_file = os.path.join(args.output, 'evaluation_results.txt')
    with open(result_file, 'w') as f:
        f.write(f"评估结果汇总 (h = {args.h}):\n\n")
        f.write(f"准确率 (Accuracy): {metrics['accuracy']:.4f}\n")
        f.write(f"精确度 (Precision): {metrics['precision']:.4f}\n")
        f.write(f"召回率 (Recall): {metrics['recall']:.4f}\n")
        f.write(f"F1分数 (F1 Score): {metrics['f1']:.4f}\n")
        f.write(f"AUC: {metrics['auc']:.4f}\n")

        f.write("\n混淆矩阵已保存到 confusion_matrix.png\n")
        f.write("ROC曲线已保存到 roc_curve.png\n")

        # 在新的模型结构中，我们不再计算不同预测提前时间的评估指标
        # 因为我们现在直接预测未来h步内是否会发生AKI
        f.write(f"\n欠采样信息:\n")
        f.write(f"提前预测步数 h = {args.h}\n")
        f.write(f"欠采样确保了类平衡，可以更准确地评估模型性能\n")
    
    print(f"评估结果已保存到 {result_file}")

if __name__ == "__main__":
    main()
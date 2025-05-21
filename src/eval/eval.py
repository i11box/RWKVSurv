import sys
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from pycox.evaluation import EvalSurv
from lifelines.utils import concordance_index
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, classification_report

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.model.model import AKIConfig, AKIPredictor, prepare_data

def load_model(model_path, config=None):
    """
    加载训练好的模型
    
    参数:
    - model_path: 模型权重文件路径
    - config: 模型配置，如果为None则使用默认配置
    
    返回:
    - 加载好的模型
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在")
    
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

def evaluate_model(model, test_data, time_steps=48, output_dir='data/results', threshold=0.8):
    """
    评估模型性能
    
    参数:
    - model: 训练好的模型
    - test_data: 测试数据DataFrame
    - time_steps: 时间步数
    - output_dir: 输出目录
    - threshold: 分类阈值，默认为0.5
    
    返回:
    - c_index: 一致性指数
    - results_df: 包含风险评分、持续时间和事件的DataFrame
    - horizon_metrics: 不同预测提前时间的评估指标
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备数据
    try:
        print("准备数据...")
        static_features, dynamic_features, targets, durations = prepare_data(test_data, time_steps)
        print(f"数据准备完成: {static_features.shape[0]} 个样本")
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
    
    # 获取模型所在的设备
    device = next(model.parameters()).device
    print(f"模型所在设备: {device}")
    
    # 将所有张量移动到相同的设备上
    static_features = static_features.to(device)
    dynamic_features = dynamic_features.to(device)
    targets = targets.to(device)
    durations = durations.to(device)
    
    # 获取风险评分
    with torch.no_grad():
        try:
            outputs = model(static_features, dynamic_features)
            
            # 处理不同的返回格式
            if isinstance(outputs, tuple) and len(outputs) >= 2:
                risk_scores, time_preds = outputs[0], outputs[1]
            else:
                risk_scores = outputs
                time_preds = None
                
            # 检查输出是否包含NaN/Inf
            check_nan_inf(risk_scores, "风险评分")
            if time_preds is not None:
                check_nan_inf(time_preds, "时间预测")
        except Exception as e:
            print(f"模型推理失败: {e}")
            raise
    
    # 打印原始形状信息
    print(f"原始 risk_scores shape: {risk_scores.shape}")
    if time_preds is not None:
        print(f"原始 time_preds shape: {time_preds.shape}")
    print(f"targets shape: {targets.shape}")
    print(f"durations shape: {durations.shape}")
    
    # 创建目标矩阵，与训练时保持一致
    batch_size, time_steps, prediction_horizon = risk_scores.shape
    target_matrix = torch.zeros_like(risk_scores)
    
    for b in range(batch_size):
        if targets[b] == 1:  # 如果该样本发生了AKI
            aki_time = int(durations[b].item())  # AKI发生的时间步
            
            for t in range(time_steps):
                if t < aki_time:
                    for h in range(min(prediction_horizon, time_steps - t)):
                        if t + h >= aki_time:
                            target_matrix[b, t, h] = 1
    
    # 计算加权二元交叉熵损失
    pos_weight = 1.0  # 与训练时保持一致
    weighted_bce_loss = F.binary_cross_entropy_with_logits(
        risk_scores, target_matrix, 
        pos_weight=torch.tensor([pos_weight], device=risk_scores.device)
    ).item()
    print(f"加权BCE损失: {weighted_bce_loss:.4f}")
    
    # 将风险评分转换为概率
    risk_probs = torch.sigmoid(risk_scores)
    
    # 聚合风险评分以计算每个样本的总体风险
    # 取每个样本的最大风险概率
    sample_risk_probs = risk_probs.max(dim=1)[0].max(dim=1)[0]
    
    # 转换为numpy数组
    risk_probs_np = risk_probs.cpu().numpy()
    target_matrix_np = target_matrix.cpu().numpy()
    sample_risk_probs_np = sample_risk_probs.cpu().numpy()
    targets_np = targets.cpu().numpy()
    durations_np = durations.cpu().numpy()
    
    # 打印聚合后的形状信息
    print(f"聚合后 risk_scores shape: {sample_risk_probs_np.shape}")
    print(f"targets shape: {targets_np.shape}")
    print(f"durations shape: {durations_np.shape}")
    
    # 将风险评分转换为二分类预测
    binary_pred = (sample_risk_probs_np > threshold).astype(int)
    
    # 创建时间步预测数组
    if time_preds is not None:
        time_pred = time_preds.cpu().numpy()
        # 聚合时间预测（取最大风险所对应的时间预测）
        # 先找到每个样本的最大风险概率的索引
        max_risk_indices = risk_probs.view(batch_size, -1).max(dim=1)[1]
        # 计算对应的时间步和预测尺度索引
        time_step_indices = max_risk_indices // prediction_horizon
        horizon_indices = max_risk_indices % prediction_horizon
        # 提取对应的时间预测
        sample_time_preds = np.array([
            time_preds[b, time_step_indices[b], horizon_indices[b]].item() 
            for b in range(batch_size)
        ])
    else:
        sample_time_preds = np.full(batch_size, -1)
    
    # 计算C-index（一致性指数）
    c_index = concordance_index(durations_np, sample_risk_probs_np, targets_np)
    
    print(f"C-index: {c_index:.4f}")
    
    # 计算混淆矩阵和分类指标
    tn, fp, fn, tp = confusion_matrix(targets_np, binary_pred).ravel()
    accuracy = accuracy_score(targets_np, binary_pred)
    precision = precision_score(targets_np, binary_pred)
    recall = recall_score(targets_np, binary_pred)
    f1 = f1_score(targets_np, binary_pred)
    
    print(f"混淆矩阵:TN: {tn}, FP: {fp} FN: {fn}, TP: {tp}")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    # 计算ROC曲线和AUC
    fpr, tpr, _ = roc_curve(targets_np, sample_risk_probs_np)
    roc_auc = auc(fpr, tpr)
    print(f"AUC: {roc_auc:.4f}")
    
    # 绘制ROC曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()
    
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'subject_id': test_data['subject_id'].values[:len(targets_np)],
        'risk_score': sample_risk_probs_np,
        'prediction': binary_pred,
        'actual': targets_np,
        'duration': durations_np,
        'actual_time': durations_np * targets_np  # 只有发生事件的样本才有实际时间
    })
    
    # 保存结果
    results_df.to_csv(os.path.join(output_dir, 'evaluation_results.csv'), index=False)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(targets_np, binary_pred, output_dir, threshold)
    
    # 绘制生存曲线
    plot_survival_curves(results_df, 3, output_dir)
    
    # 评估不同预测提前时间的性能
    # 定义要评估的预测提前时间（horizon）
    horizons_to_evaluate = [6, 8, 10, 12]
    horizon_metrics = evaluate_prediction_horizons(
        risk_probs_np, target_matrix_np, targets_np, durations_np, 
        horizons_to_evaluate, threshold, output_dir
    )
    
    return c_index, results_df, horizon_metrics

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
        static_features, dynamic_features, targets, durations = prepare_data(test_data, args.time_steps)
        
        # 获取特征维度
        static_dim = static_features.shape[1]  # 静态特征维度
        dynamic_dim = dynamic_features.shape[2]  # 动态特征维度
        time_steps = dynamic_features.shape[1]  # 时间步数
        
        print(f"特征维度: static_dim={static_dim}, dynamic_dim={dynamic_dim}, time_steps={time_steps}")
        
        # 创建模型配置
        config = AKIConfig(
            static_dim=static_dim,
            dynamic_dim=dynamic_dim,
            ctx_len=time_steps,
            embed_dim=128,
            n_layer=3,
            n_head=4
        )
        
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
    c_index, results_df, horizon_metrics = evaluate_model(model, test_data, args.time_steps, args.output, args.threshold)
    
    # 绘制生存曲线
    plot_survival_curves(results_df, args.groups, args.output)
    
    # 保存评估结果
    result_file = os.path.join(args.output, 'evaluation_results.txt')
    with open(result_file, 'w') as f:
        f.write(f"C-index: {c_index:.4f}\n\n")
        
        # 保存不同预测提前时间的评估指标
        f.write("不同预测提前时间的评估指标：\n")
        
        # 确保 horizon_metrics 是一个字典
        if not isinstance(horizon_metrics, dict):
            f.write("警告: 无法获取预测提前时间的评估指标 (无效的格式)\n")
            print(f"警告: horizon_metrics 的类型是 {type(horizon_metrics)}, 预期是字典")
            if horizon_metrics is not None:
                print(f"horizon_metrics 内容: {horizon_metrics}")
        else:
            for horizon, metrics in horizon_metrics.items():
                try:
                    if not isinstance(metrics, dict):
                        f.write(f"\n预测提前时间: {horizon} 小时 - 警告: 指标格式无效\n")
                        print(f"警告: 预测提前时间 {horizon} 的指标不是字典格式: {type(metrics)}")
                        continue
                        
                    f.write(f"\n预测提前时间: {horizon} 小时\n")
                    # 使用安全的字典访问方式
                    metrics_data = {
                        'accuracy': metrics.get('accuracy', 0),
                        'precision': metrics.get('precision', 0),
                        'recall': metrics.get('recall', 0),
                        'f1': metrics.get('f1', 0),
                        'auc': metrics.get('auc', 0),
                        'tn': metrics.get('tn', 0),
                        'fp': metrics.get('fp', 0),
                        'fn': metrics.get('fn', 0),
                        'tp': metrics.get('tp', 0)
                    }
                    
                    f.write(f"  - 准确率: {metrics_data['accuracy']:.4f}\n")
                    f.write(f"  - 精确率: {metrics_data['precision']:.4f}\n")
                    f.write(f"  - 召回率: {metrics_data['recall']:.4f}\n")
                    f.write(f"  - F1分数: {metrics_data['f1']:.4f}\n")
                    f.write(f"  - AUC: {metrics_data['auc']:.4f}\n")
                    f.write(f"  - 混淆矩阵: TN={metrics_data['tn']}, FP={metrics_data['fp']}, "
                          f"FN={metrics_data['fn']}, TP={metrics_data['tp']}\n")
                except Exception as e:
                    f.write(f"\n预测提前时间: {horizon} 小时 - 处理指标时出错: {str(e)}\n")
                    print(f"处理预测提前时间 {horizon} 的指标时出错: {e}")
    
    print(f"评估结果已保存到 {result_file}")

if __name__ == "__main__":
    main()
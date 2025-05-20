import sys
import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pycox.evaluation import EvalSurv
from lifelines.utils import concordance_index
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.model.model import AKIConfig, AKIPredictor, prepare_data

def load_model(model_path):
    """
    加载训练好的模型
    
    参数:
    - model_path: 模型权重文件路径
    
    返回:
    - 加载好的模型
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在")
    
    # 使用weights_only=False参数加载模型，允许加载自定义类
    model = torch.load(model_path, weights_only=False)
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

def evaluate_model(model, test_data, time_steps=48, output_dir='data/results', threshold=0.5):
    """
    评估模型性能
    
    参数:
    - model: 训练好的模型
    - test_data: 测试数据DataFrame
    - time_steps: 时间步数
    - output_dir: 输出目录
    - threshold: 分类阈值，默认为0.5
    - output_dir: 输出目录
    
    返回:
    - c_index: 一致性指数
    - results_df: 包含风险评分、持续时间和事件的DataFrame
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备数据
    static_features, dynamic_features, targets, durations = prepare_data(test_data, time_steps)
    
    # 获取模型所在的设备
    device = next(model.parameters()).device
    print(f"模型所在设备: {device}")
    
    # 将所有张量移动到相同的设备上
    static_features = static_features.to(device)
    dynamic_features = dynamic_features.to(device)
    targets = targets.to(device)
    durations = durations.to(device)
    
    # 获取风险评分（暂时忽略时间步预测）
    with torch.no_grad():
        risk_scores, _, _ = model(static_features, dynamic_features)
    
    # 转换为numpy数组并确保是一维的
    risk_scores = risk_scores.cpu().numpy().flatten()  # 确保是一维
    targets = targets.cpu().numpy().flatten()  # 确保是一维
    durations = durations.cpu().numpy().flatten()  # 确保是一维
    
    # 打印形状信息以进行调试
    print(f"risk_scores shape: {risk_scores.shape}")
    print(f"targets shape: {targets.shape}")
    print(f"durations shape: {durations.shape}")
    
    # 将风险评分转换为二分类预测
    binary_pred = (risk_scores > threshold).astype(int)
    
    # 创建一个全为-1的时间步预测数组（表示暂时不预测时间步）
    time_pred = np.full_like(risk_scores, -1)  
    
    # 计算C-index（一致性指数）
    c_index = concordance_index(durations, risk_scores, targets)
    
    print(f"C-index: {c_index:.4f}")
    
    # 评估时间步预测的准确性（只对实际发生AKI的样本）
    mask_actual_aki = targets == 1
    if mask_actual_aki.sum() > 0:
        # 对于实际发生AKI的样本，计算时间步预测的误差
        # 暂时禁用时间步预测的评估
        # mask_both = (targets == 1) & (time_pred != -1)
        # if mask_both.sum() > 0:
        #     # 计算平均误差和中位数误差
        #     errors = np.abs(time_pred[mask_both] - durations[mask_both])
        #     mean_error = np.mean(errors)
        #     median_error = np.median(errors)
        #     print(f"时间步预测的平均误差: {mean_error:.4f}")
        #     print(f"时间步预测的中位数误差: {median_error:.4f}")
        # else:
        #     print("没有实际发生AKI并且预测也发生AKI的样本，无法计算时间步预测的误差")
        # 使用二分类预测而不是时间步预测
        fn_rate = np.sum(mask_actual_aki & (binary_pred == 0)) / np.sum(mask_actual_aki)
        print(f"假负例率（实际发生AKI但预测不发生的比例）: {fn_rate:.4f}")
    
    # 计算预测为“发生AKI”但实际不发生的比例（即假正例率）
    mask_actual_no_aki = targets == 0
    if mask_actual_no_aki.sum() > 0:
        # 使用二分类预测而不是时间步预测
        fp_rate = np.sum(mask_actual_no_aki & (binary_pred == 1)) / np.sum(mask_actual_no_aki)
        print(f"假正例率（实际不发生AKI但预测发生的比例）: {fp_rate:.4f}")
    
    # 创建用于可视化的DataFrame
    results_df = pd.DataFrame({
        'risk_score': risk_scores,
        # 'predicted_time': time_pred,  # 暂时禁用时间步预测
        'actual_time': durations,
        'event': targets
    })
    
    # 生成混淆矩阵和其他评估指标
    cm, report = plot_confusion_matrix(targets, risk_scores, output_dir, threshold)
    
    # 暂时禁用时间步预测的可视化
    # plt.figure(figsize=(10, 6))
    # 
    # # 只对实际发生AKI并且预测也发生AKI的样本进行可视化
    # mask_both = (targets == 1) & (time_pred != -1)
    # if mask_both.sum() > 0:
    #     plt.scatter(durations[mask_both], time_pred[mask_both], alpha=0.5)
    #     
    #     # 添加对角线（完美预测线）
    #     min_time = min(np.min(durations[mask_both]), np.min(time_pred[mask_both]))
    #     max_time = max(np.max(durations[mask_both]), np.max(time_pred[mask_both]))
    #     plt.plot([min_time, max_time], [min_time, max_time], 'r--')
    #     
    #     plt.xlabel('实际AKI发生时间步')
    #     plt.ylabel('预测AKI发生时间步')
    #     plt.title('AKI发生时间步预测')
    #     plt.grid(True, alpha=0.3)
    #     
    #     # 保存图像
    #     time_pred_file = os.path.join(output_dir, 'time_prediction.png')
    #     plt.savefig(time_pred_file)
    #     plt.close()
    #     print(f"时间步预测图已保存到 {time_pred_file}")
    # else:
    #     print("没有实际发生AKI并且预测也发生AKI的样本，无法生成时间步预测图")
    
    return c_index, results_df

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
    parser.add_argument('--data', type=str, default='data/aki_hypertension_data_eval.csv', help='测试数据路径')
    parser.add_argument('--time_steps', type=int, default=48, help='时间步数')
    parser.add_argument('--output', type=str, default='data/results', help='输出结果目录')
    parser.add_argument('--groups', type=int, default=3, help='风险组数量')
    parser.add_argument('--threshold', type=float, default=0.5, help='分类阈值，默认为0.5，用于生成混淆矩阵')
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
    
    # 加载模型
    try:
        model = load_model(args.model)
        print(f"成功加载模型: {args.model}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 评估模型
    c_index, results_df = evaluate_model(model, test_data, args.time_steps, args.output, args.threshold)
    
    # 绘制生存曲线
    plot_survival_curves(results_df, args.groups, args.output)
    
    # 保存评估结果
    result_file = os.path.join(args.output, 'evaluation_results.txt')
    with open(result_file, 'w') as f:
        f.write(f"C-index: {c_index:.4f}\n")
    
    print(f"评估结果已保存到 {result_file}")

if __name__ == "__main__":
    main()
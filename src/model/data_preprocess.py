#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AKI高血压数据预处理脚本

该脚本对AKI高血压数据集进行预处理，包括：
1. 静态特征处理
2. 目标变量处理和特征选择
3. 动态特征处理
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import optuna
from optuna.samplers import TPESampler
from xgboost import XGBClassifier
import argparse
import os


def process_static_features(data):
    """
    处理静态特征
    
    参数:
    - data: 原始数据DataFrame
    
    返回:
    - processed_data: 处理后的DataFrame，只包含处理后的静态特征
    """
    print("处理静态特征...")
    
    # 复制数据，避免修改原始数据
    df = data.copy()
    
    # 1. 去掉所有subject_id重复的记录，只保留一个
    print(f"去重前记录数: {len(df)}")
    df = df.drop_duplicates(subset=['subject_id'], keep='first')
    print(f"去重后记录数: {len(df)}")
    
    # 2. 删除stay_id列
    df = df.drop('stay_id', axis=1)
    
    # 3. 提取静态特征列
    static_cols = ['subject_id', 'gender', 'age', 'marital_status', 'language', 'ethnicity', 'curr_service']
    static_df = df[static_cols + ['aki_time']]
    
    # 4. 处理空值：所有静态特征的空值按众数处理
    for col in static_cols[1:]:  # 跳过subject_id
        if static_df[col].isnull().sum() > 0:
            mode_value = static_df[col].mode()[0]
            static_df[col] = static_df[col].fillna(mode_value)
            print(f"列 {col} 的空值已用众数 {mode_value} 填充")
    
    # 5. 处理性别：gender按M为1，F为0处理
    static_df['gender'] = static_df['gender'].replace({'M': 1, 'F': 0}).astype(int)
    
    # 6. 处理年龄：对age进行极大极小归一化
    age_scaler = MinMaxScaler()
    static_df['age'] = age_scaler.fit_transform(static_df[['age']])
    
    # 7. 去掉所有年龄为0和为1的记录（极端值）
    print(f"去除极端年龄前记录数: {len(static_df)}")
    static_df = static_df[(static_df['age'] > 0) & (static_df['age'] < 1)]
    print(f"去除极端年龄后记录数: {len(static_df)}")
    
    # 8. 独热编码：对marital_status、language、ethnicity、curr_service进行独热编码
    categorical_cols = ['marital_status', 'language', 'ethnicity', 'curr_service']
    
    # 对每个分类特征进行独热编码
    for col in categorical_cols:
        # 创建OneHotEncoder
        encoder = OneHotEncoder(handle_unknown='ignore')
        
        # 拟合并转换，确保结果是密集数组
        encoded_features = encoder.fit_transform(static_df[[col]]).toarray()
        
        # 创建编码后的列名
        feature_names = [f"{col}_{val}" for val in encoder.categories_[0]]
        
        # 创建包含编码特征的DataFrame
        encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=static_df.index)
        
        # 将编码特征添加到原始DataFrame
        static_df = pd.concat([static_df, encoded_df], axis=1)
        
        # 删除原始分类列
        static_df = static_df.drop(col, axis=1)
    
    print(f"独热编码后的静态特征数量: {len(static_df.columns) - 2}")  # 减去subject_id和aki_time
    
    return static_df


def select_features_with_xgboost(static_df):
    """
    使用XGBoost进行特征选择
    
    参数:
    - static_df: 处理后的静态特征DataFrame
    
    返回:
    - selected_features_df: 选择后的特征DataFrame
    """
    print("使用XGBoost进行特征选择...")
    
    # 复制数据，避免修改原始数据
    df = static_df.copy()
    
    # 1. 创建目标变量aki_status
    df['aki_status'] = np.where(df['aki_time'] == -1, 0, 1)
    print(f"正样本(AKI)数量: {df['aki_status'].sum()}")
    print(f"负样本(无AKI)数量: {len(df) - df['aki_status'].sum()}")
    
    # 2. 对负样本进行欠采样，使负样本数量为正样本数量的1.5倍
    positive_samples = df[df['aki_status'] == 1]
    negative_samples = df[df['aki_status'] == 0]
    
    # 计算需要保留的负样本数量
    n_positive = len(positive_samples)
    n_negative_target = int(n_positive * 1.5)
    
    # 如果负样本数量大于目标数量，则进行欠采样
    if len(negative_samples) > n_negative_target:
        negative_samples = negative_samples.sample(n=n_negative_target, random_state=3407)
        print(f"对负样本进行欠采样，保留 {n_negative_target} 条记录")
    
    # 合并正样本和采样后的负样本
    balanced_df = pd.concat([positive_samples, negative_samples])
    print(f"平衡后的数据集大小: {len(balanced_df)}")
    
    # 3. 准备特征和目标变量
    X = balanced_df.drop(['subject_id', 'aki_time', 'aki_status'], axis=1)
    y = balanced_df['aki_status']
    
    # 4. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=3407, stratify=y
    )
    
    print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
    print(f"训练集类别分布: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"测试集类别分布: {dict(zip(*np.unique(y_test, return_counts=True)))}")
    
    # 5. 使用Optuna优化XGBoost超参数
    print("\n使用Optuna优化XGBoost超参数...")
    
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'random_state': 3407,
            'n_jobs': -1,
            'eval_metric': 'logloss'
        }
        
        # 使用交叉验证评估参数
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=3407)
        
        # 计算F1分数
        f1_scores = cross_val_score(
            XGBClassifier(**params),
            X_train, y_train,
            cv=cv,
            scoring='f1',
            n_jobs=-1
        )
        mean_f1 = f1_scores.mean()
        
        # 计算准确率
        accuracy_scores = cross_val_score(
            XGBClassifier(**params),
            X_train, y_train,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1
        )
        mean_accuracy = accuracy_scores.mean()
        
        # 计算ROC-AUC分数 (仅作为参考)
        auc_scores = cross_val_score(
            XGBClassifier(**params),
            X_train, y_train,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1
        )
        mean_auc = auc_scores.mean()
        
        # 组合分数，优先考虑F1分数，其次是准确率
        # 确保F1分数和准确率都高于0.5
        if mean_f1 < 0.5 or mean_accuracy < 0.5:
            # 如果F1或准确率低于0.5，添加惩罚
            f1_penalty = 10.0 * max(0, 0.5 - mean_f1)  # F1惩罚权重更大
            acc_penalty = 5.0 * max(0, 0.5 - mean_accuracy)
            combined_score = mean_f1 - f1_penalty - acc_penalty  # 以F1为基础
        else:
            # 当基本要求满足时，主要关注F1分数
            # F1权重为0.7，准确率权重为0.3
            combined_score = 0.7 * mean_f1 + 0.3 * mean_accuracy
        
        # 打印评估结果（可选）
        trial.set_user_attr('f1', mean_f1)
        trial.set_user_attr('accuracy', mean_accuracy)
        trial.set_user_attr('auc', mean_auc)
        trial.set_user_attr('combined_score', combined_score)
        
        return combined_score
    
    # 创建Optuna study并优化
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=3407))
    study.optimize(objective, n_trials=100, n_jobs=-1)
    
    # 打印最佳参数和分数
    best_params = study.best_params
    best_trial = study.best_trial
    print(f"\n最佳参数: {best_params}")
    print(f"最佳F1分数: {best_trial.user_attrs['f1']:.4f}")
    print(f"最佳准确率: {best_trial.user_attrs['accuracy']:.4f}")
    print(f"最佳AUC分数: {best_trial.user_attrs['auc']:.4f}")
    
    # 使用最佳参数训练最终模型
    print("\n使用最佳参数训练最终模型...")
    model = XGBClassifier(**best_params, random_state=3407)
    model.fit(X_train, y_train)
    
    # 6. 在测试集上进行预测
    print("评估模型性能...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # 打印评估结果
    print("\n原始测试集评估结果:")
    print(f"- 准确率(Accuracy): {accuracy:.4f}")
    print(f"- 精确率(Precision): {precision:.4f}")
    print(f"- 召回率(Recall): {recall:.4f}")
    print(f"- F1分数: {f1:.4f}")
    print(f"- AUC-ROC: {roc_auc:.4f}")
    
    # 对测试集进行欠采样
    print("\n对测试集进行欠采样...")
    print(f"欠采样前测试集类别分布: {dict(zip(*np.unique(y_test, return_counts=True)))}")
    
    # 对测试集进行欠采样，使负样本数量为正样本数量的1.5倍
    X_test_pos = X_test[y_test == 1]  # 正样本（AKI）
    y_test_pos = y_test[y_test == 1]
    X_test_neg = X_test[y_test == 0]  # 负样本（无AKI）
    y_test_neg = y_test[y_test == 0]
    
    # 计算需要保留的负样本数量
    n_positive = len(X_test_pos)
    n_negative_target = int(n_positive * 1.5)
    
    # 如果负样本数量大于目标数量，则进行欠采样
    if len(X_test_neg) > n_negative_target:
        # 随机选择负样本
        neg_indices = np.random.choice(len(X_test_neg), n_negative_target, replace=False)
        X_test_neg = X_test_neg.iloc[neg_indices]
        y_test_neg = y_test_neg.iloc[neg_indices]
        print(f"对负样本进行欠采样，保留 {n_negative_target} 条记录")
    
    # 合并正样本和采样后的负样本
    X_test_balanced = pd.concat([X_test_pos, X_test_neg])
    y_test_balanced = pd.concat([y_test_pos, y_test_neg])
    
    # 打印欠采样后的类别分布
    print(f"欠采样后测试集类别分布: {dict(zip(*np.unique(y_test_balanced, return_counts=True)))}")
    
    # 在欠采样测试集上进行预测
    y_pred_balanced = model.predict(X_test_balanced)
    y_pred_proba_balanced = model.predict_proba(X_test_balanced)[:, 1]
    
    # 计算评估指标
    accuracy_balanced = accuracy_score(y_test_balanced, y_pred_balanced)
    precision_balanced = precision_score(y_test_balanced, y_pred_balanced)
    recall_balanced = recall_score(y_test_balanced, y_pred_balanced)
    f1_balanced = f1_score(y_test_balanced, y_pred_balanced)
    roc_auc_balanced = roc_auc_score(y_test_balanced, y_pred_proba_balanced)
    
    # 打印欠采样测试集评估结果
    print("\n欠采样测试集评估结果 (仅用于分析):")
    print(f"- 准确率(Accuracy): {accuracy_balanced:.4f}")
    print(f"- 精确率(Precision): {precision_balanced:.4f}")
    print(f"- 召回率(Recall): {recall_balanced:.4f}")
    print(f"- F1分数: {f1_balanced:.4f}")
    print(f"- AUC-ROC: {roc_auc_balanced:.4f}")
    
    # 绘制ROC曲线
    plt.figure(figsize=(8, 6))
    
    # 原始测试集的ROC曲线
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'原始测试集 (AUC = {roc_auc:.2f})')
    
    # 欠采样测试集的ROC曲线
    fpr_balanced, tpr_balanced, _ = roc_curve(y_test_balanced, y_pred_proba_balanced)
    plt.plot(fpr_balanced, tpr_balanced, color='green', lw=2, label=f'欠采样测试集 (AUC = {roc_auc_balanced:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    print("\nROC曲线已保存为 'roc_curve.png'")
    
    # 5. 获取特征重要性
    feature_importance = model.feature_importances_
    feature_names = X.columns
    
    # 创建特征重要性DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    # 打印age特征的重要性
    if 'age' in importance_df['feature'].values:
        age_importance = importance_df[importance_df['feature'] == 'age'].iloc[0]
        rank = importance_df.index.get_loc(age_importance.name) + 1  # 获取排名（从1开始）
        print(f"\n特征'age'的重要性:")
        print(f"- 重要性分数: {age_importance['importance']:.6f}")
        print(f"- 在所有特征中的排名: {rank}/{len(importance_df)}")
        print(f"- 重要性百分位数: {((len(importance_df) - rank) / len(importance_df) * 100):.1f}%")
    else:
        print("\n警告: 未找到'age'特征")
    
    # 绘制特征重要性图
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=importance_df.head(20))
    plt.title('Top 20 Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("特征重要性图已保存为 'feature_importance.png'")
    
    # 6. 根据累计重要性达到95%选择特征
    importance_df = importance_df.sort_values('importance', ascending=False)
    importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
    importance_df['cumulative_importance'] = importance_df['cumulative_importance'] / importance_df['importance'].sum()
    
    # 选择累计重要性达到95%的特征
    important_features = importance_df[importance_df['cumulative_importance'] <= 0.95]['feature'].tolist()
    
    print(f"选择的重要特征数量: {len(important_features)}")
    print(f"重要特征: {important_features}")
    
    # 7. 创建包含重要特征的DataFrame
    selected_features_df = static_df[['subject_id', 'aki_time'] + important_features]
    
    return selected_features_df, important_features


def process_dynamic_features(data, static_df):
    """
    处理动态特征
    
    参数:
    - data: 原始数据DataFrame
    - static_df: 处理后的静态特征DataFrame
    
    返回:
    - final_df: 包含处理后的静态和动态特征的DataFrame
    """
    print("处理动态特征...")
    
    # 复制数据，避免修改原始数据
    df = data.copy()
    
    # 1. 获取所有动态特征列
    all_columns = df.columns.tolist()
    static_cols = ['stay_id', 'subject_id', 'gender', 'age', 'marital_status', 'language', 'ethnicity', 'curr_service', 'aki_time']
    dynamic_cols = [col for col in all_columns if col not in static_cols]
    
    # 2. 按特征类型分组
    feature_types = set()
    for col in dynamic_cols:
        # 提取特征类型（去掉_t和数字）
        if '_t' in col:
            feature_type = col.split('_t')[0]
            feature_types.add(feature_type)
    
    # 去除admission_weight和temperature特征
    feature_types = {ft for ft in feature_types if ft not in ['admission_weight', 'temperature']}
    
    print(f"动态特征类型数量: {len(feature_types)}")
    print(f"动态特征类型: {feature_types}")
    
    # 3. 创建一个新的DataFrame，只包含subject_id和静态特征
    subject_ids = static_df['subject_id'].tolist()
    filtered_df = df[df['subject_id'].isin(subject_ids)].copy()
    
    # 3. 对每种特征类型进行处理
    for feature_type in feature_types:
        print(f"处理特征类型: {feature_type}")
        
        # 获取该特征类型的所有列
        type_cols = [col for col in dynamic_cols if col.startswith(feature_type + '_t')]
        
        # 排序列，确保按时间顺序
        type_cols.sort(key=lambda x: int(x.split('_t')[1].split('_')[0]) if '_' in x.split('_t')[1] else int(x.split('_t')[1]))
        
        # 创建一个新的DataFrame，用于存储归一化后的值
        normalized_df = filtered_df[['subject_id']].copy()
        
        # 对每个时间点的特征分别进行归一化
        for col in type_cols:
            # 复制原始值
            normalized_df[col] = filtered_df[col].copy()
            
            # 获取非空值的数据点
            non_null_mask = ~filtered_df[col].isnull()
            
            # 如果有非空值，则进行归一化
            if non_null_mask.sum() > 0:
                # 计算当前列的最小值和范围
                col_min = filtered_df.loc[non_null_mask, col].min()
                col_range = filtered_df.loc[non_null_mask, col].max() - col_min
                
                # 避免除以零
                if col_range > 0:
                    normalized_df.loc[non_null_mask, col] = (filtered_df.loc[non_null_mask, col] - col_min) / col_range
        
        # 对每个病人进行线性插值
        for subject_id in normalized_df['subject_id'].unique():
            subject_mask = normalized_df['subject_id'] == subject_id
            subject_data = normalized_df.loc[subject_mask, type_cols]
            
            # 线性插值
            interpolated_data = subject_data.interpolate(method='linear', axis=1, limit_direction='both')
            normalized_df.loc[subject_mask, type_cols] = interpolated_data
        
        # 将归一化和插值后的特征添加到filtered_df
        for col in type_cols:
            filtered_df[col] = normalized_df[col]
    
    # 5. 合并静态特征和处理后的动态特征
    final_df = pd.merge(static_df, filtered_df[['subject_id'] + dynamic_cols], on='subject_id', how='left')
    
    return final_df


def remove_rows_with_missing_values(df, columns=None, max_missing_pct=100.0, inplace=False):
    """
    移除包含空值的行
    
    参数:
    - df: 输入DataFrame
    - columns: 要检查的列名列表，如果为None则检查所有列
    - max_missing_pct: 允许的最大缺失值百分比，超过此值将引发异常
    - inplace: 是否原地修改DataFrame
    
    返回:
    - df_cleaned: 处理后的DataFrame（如果inplace=True则返回None）
    
    示例:
    >>> df = pd.DataFrame({'A': [1, 2, None], 'B': [4, None, 6]})
    >>> df_cleaned = remove_rows_with_missing_values(df)
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime
    
    # 输入验证
    if not isinstance(df, pd.DataFrame):
        raise TypeError("输入必须是一个pandas DataFrame")
        
    if df.empty:
        print("警告: 输入DataFrame为空")
        return df if not inplace else None
    
    # 记录开始时间
    start_time = datetime.now()
    print(f"\n[{start_time.strftime('%Y-%m-%d %H:%M:%S')}] 开始处理缺失值...")
    
    # 确定要检查的列
    columns_to_check = columns if columns is not None else df.columns
    if not isinstance(columns_to_check, (list, pd.Index)):
        columns_to_check = [columns_to_check]
    
    # 检查列是否存在
    missing_cols = set(columns_to_check) - set(df.columns)
    if missing_cols:
        raise ValueError(f"以下列不存在于DataFrame中: {missing_cols}")
    
    # 记录处理前的行数和缺失值统计
    n_rows_before = len(df)
    missing_stats = df[columns_to_check].isnull().sum()
    missing_cols = missing_stats[missing_stats > 0]
    
    if not missing_cols.empty:
        print("\n各列缺失值数量:")
        for col, count in missing_cols.items():
            pct = (count / n_rows_before) * 100
            print(f"  - {col}: {count} 行 ({pct:.2f}%)")
    
    # 计算并检查缺失值百分比
    total_missing = missing_stats.sum()
    missing_pct = (total_missing / (n_rows_before * len(columns_to_check))) * 100
    
    if missing_pct > max_missing_pct:
        raise ValueError(f"缺失值比例 ({missing_pct:.2f}%) 超过最大允许值 ({max_missing_pct}%)")
    
    # 移除包含空值的行
    if inplace:
        df.dropna(subset=columns_to_check, inplace=True)
        df_cleaned = df
    else:
        df_cleaned = df.dropna(subset=columns_to_check)
    
    # 计算移除了多少行
    n_removed = n_rows_before - len(df_cleaned)
    
    # 记录处理时间
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # 打印统计信息
    if n_removed > 0:
        print(f"\n[{end_time.strftime('%Y-%m-%d %H:%M:%S')}] 完成! 处理耗时: {duration:.2f}秒")
        print(f"移除了 {n_removed} 行包含空值的数据 ({(n_removed/n_rows_before)*100:.2f}%)")
        print(f"处理前行数: {n_rows_before}, 处理后行数: {len(df_cleaned)}")
        
        # 检查是否还有空值
        remaining_missing = df_cleaned[columns_to_check].isnull().sum().sum()
        if remaining_missing > 0:
            print("警告: 数据中仍然存在空值")
            print("各列剩余空值数量:")
            print(df_cleaned.isnull().sum()[df_cleaned.isnull().sum() > 0])
    else:
        print(f"\n[{end_time.strftime('%Y-%m-%d %H:%M:%S')}] 完成! 未发现包含空值的行")
    
    return None if inplace else df_cleaned


def main(input_file, output_file):
    """
    主函数
    
    参数:
    - input_file: 输入文件路径
    - output_file: 输出文件路径
    """
    print(f"加载数据: {input_file}")
    data = pd.read_csv(input_file)
    print(f"原始数据大小: {data.shape}")
    
    # 1. 处理静态特征
    static_df = process_static_features(data)
    
    # 2. 使用XGBoost进行特征选择
    selected_static_df, important_features = select_features_with_xgboost(static_df)
    
    # 3. 处理动态特征
    final_df = process_dynamic_features(data, selected_static_df)
    
    # 4. 保存处理后的数据
    print(f"保存处理后的数据: {output_file}")
    final_df.to_csv(output_file, index=False)
    print(f"处理后的数据大小: {final_df.shape}")
    print("数据预处理完成!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AKI高血压数据预处理')
    parser.add_argument('--input', type=str, default='data/aki_hypertension_data.csv',
                        help='输入文件路径')
    parser.add_argument('--output', type=str, default='data/aki_hypertension_data_processed.csv',
                        help='输出文件路径')
    args = parser.parse_args()
    
    main(args.input, args.output)
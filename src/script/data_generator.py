import pandas as pd
import numpy as np
import os

def generate_mock_data(n_samples=200, output_path='data/generated_data.csv'):
    """
    生成与mock_data.csv具有相同特征维度的模拟数据
    
    参数:
    - n_samples: 要生成的样本数量
    - output_path: 输出CSV文件的路径
    
    返回:
    - 生成的DataFrame
    """
    # 创建ID列
    ids = np.arange(1, n_samples + 1)
    
    # 生成静态特征
    static_feature1 = np.random.randint(20, 90, size=n_samples)  # 年龄范围20-90
    static_feature2 = np.random.randint(0, 2, size=n_samples)    # 二元特征(0或1)
    
    # 生成动态特征基础值
    base_dynamic_feature1 = np.random.uniform(0.5, 1.0, size=n_samples)  # 基础值
    base_dynamic_feature2 = np.random.uniform(0.8, 1.5, size=n_samples)  # 基础值
    
    # 为每个时间点生成动态特征
    dynamic_features = {}
    for t in range(1, 5):  # t1到t4
        # 随着时间的推移，动态特征可能会增加
        trend_factor = 1.0 + 0.1 * t  # 每个时间步增加10%
        noise = np.random.normal(0, 0.05, size=n_samples)  # 添加一些随机噪声
        
        dynamic_features[f'动态特征1_t{t}'] = base_dynamic_feature1 * trend_factor + noise
        dynamic_features[f'动态特征2_t{t}'] = base_dynamic_feature2 * trend_factor + noise
    
    # 确保动态特征都是正数
    for key in dynamic_features:
        dynamic_features[key] = np.maximum(0.1, dynamic_features[key])
    
    # 生成AKI发生时间点
    # 约40%的样本不发生AKI，60%的样本在t1-t4期间发生
    aki_time = np.random.choice(['未发生', '1', '2', '3', '4'], size=n_samples, 
                              p=[0.4, 0.1, 0.2, 0.2, 0.1])
    
    # 创建DataFrame
    data = {
        'ID': ids,
        '静态特征1': static_feature1,
        '静态特征2': static_feature2
    }
    
    # 添加动态特征
    data.update(dynamic_features)
    
    # 添加AKI发生时间点
    data['AKI发生时间点'] = aki_time
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 保存到CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f'已生成{n_samples}个样本，保存到{output_path}')
    return df

# 如果直接运行此脚本，则生成数据
if __name__ == '__main__':
    generate_mock_data(200, 'data/generated_data.csv')

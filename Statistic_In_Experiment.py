import numpy as np
import pandas as pd
from scipy import stats


def calculate_f2(group):
	"""计算分组的F2分数"""
	# 真实标签：attack_ratio > 0表示被攻击
	y_true = (group['attack_ratio'] > 0).astype(int)
	# 预测标签：detected列（0表示未检测到，1表示检测到）
	y_pred = group['successful_detection'].astype(int)
	
	# 计算混淆矩阵
	tp = ((y_true == 1) & (y_pred == 1)).sum()
	fp = ((y_true == 0) & (y_pred == 1)).sum()
	fn = ((y_true == 1) & (y_pred == 0)).sum()
	
	# 处理除以零的情况
	precision = tp / (tp + fp) if (tp + fp) > 0 else 0
	recall = tp / (tp + fn) if (tp + fn) > 0 else 0
	
	# 计算F2分数（beta=2，召回率权重是精确率的2倍）
	f2 = (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) != 0 else 0
	return f2


def bootstrap_confidence_interval_df(df, confidence_level=0.95, num_bootstrap_samples=1000):
	"""
	运用自助法计算 DataFrame 每列的置信区间，结果存于 DataFrame 中。

	参数:
	df (pandas.DataFrame): 输入的 DataFrame。
	confidence_level (float): 置信水平，默认值为 0.95。
	num_bootstrap_samples (int): 自助采样的次数，默认值为 1000。

	返回:
	pandas.DataFrame: 包含每列置信区间的 DataFrame，索引为 'lower_bound' 和 'upper_bound'。
	"""
	result = {}
	for col in df.columns:
		data = df[col].values
		bootstrap_means = []
		for _ in range(num_bootstrap_samples):
			bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
			bootstrap_mean = np.mean(bootstrap_sample)
			bootstrap_means.append(bootstrap_mean)
		
		alpha = 1 - confidence_level
		lower_percentile = alpha / 2 * 100
		upper_percentile = (1 - alpha / 2) * 100
		lower_bound = np.percentile(bootstrap_means, lower_percentile)
		upper_bound = np.percentile(bootstrap_means, upper_percentile)
		result[col] = [lower_bound, upper_bound]
	
	return pd.DataFrame(result, index=['lower_bound', 'upper_bound']).T

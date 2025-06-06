#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：PoisonCatcher
@File ：Real_Data_Process.py
@Author ：SLS
@Date ：25.01.21
"""

import numpy as np
import pandas as pd
from scipy import stats
from collections import Counter


def laplace_mean_estimation(privatized_x):
	"""
	对使用Laplace噪声的LDP方式进行数据均值估计

	参数:
	privatized_x - 原始laplace扰动数据

	返回:
	统计数据均值
	"""
	# 计算均值估计
	return np.mean(privatized_x)


def grr_frequency_estimation(reported_values, domain, epsilon):
	"""
	使用 GRR 机制进行频数估计。

	参数:
	reported_values (list of str): 使用 GRR 机制扰动后的数据。
	domain (list): 数据的取值域（所有可能的值）。
	epsilon (float): 隐私预算。

	返回:
	dict: 每个值的估计频数分布。
	"""
	n = len(reported_values)
	domain_size = len(domain)
	p = np.exp(epsilon) / (np.exp(epsilon) + domain_size - 1)
	q = 1 / (np.exp(epsilon) + domain_size - 1)
	
	# 统计每个值的观察频数
	observed_counts = Counter(reported_values)
	
	# 频数估计
	estimated_frequencies = {}
	for value in domain:
		observed_count = observed_counts.get(value, 0)
		estimated_frequency = (observed_count - n * q) / (p - q)
		estimated_frequency = max(0, estimated_frequency)  # 防止负值
		estimated_frequencies[value] = estimated_frequency
	
	# 归一化为频率分布
	total_estimated = sum(estimated_frequencies.values())
	for key in estimated_frequencies:
		estimated_frequencies[key] /= total_estimated
	
	return estimated_frequencies


def calculate_features_statistics(waiting_attacked_dataset, discrete_attribute, epsilon):
	# 对当前时刻数据集中每个属性计算统计结果
	statistics_dict = {}  # 创建一个空字典用于存储统计结果
	for column in waiting_attacked_dataset.columns:
		if column not in discrete_attribute:
			statistics_dict[column] = laplace_mean_estimation(waiting_attacked_dataset[column])
		else:
			domain = waiting_attacked_dataset[column].unique().tolist()
			statistics_dict[column] = grr_frequency_estimation(waiting_attacked_dataset[column], domain, epsilon)
	return statistics_dict


def calculate_ft(data, count, epsilon, confidence_level, discrete_attrs):
	alpha = {}
	
	# 计算离散属性alpha
	for i in discrete_attrs:
		matching_columns = [col for col in data.columns if col.startswith(i)]
		denominator = (np.exp(epsilon) - 1) * np.sqrt(np.pi * count * (1 - confidence_level))
		numerator = 2 * (np.exp(epsilon) + len(matching_columns) - 2)
		alpha[i] = numerator / denominator
	
	# 计算 laplace alpha
	alpha_lap = np.sqrt(2) * 2 / (epsilon * np.sqrt(count * (1 - confidence_level)))
	
	# 更新 alpha 字典，将连续属性的 alpha 设置为 alpha_lap
	alpha.update({col: alpha_lap for col in data.columns if col not in discrete_attrs})
	
	return pd.DataFrame.from_dict(alpha, orient='index')


def compute_bias(discrete_attrs, origin_attack_bias):
	origin_LDP_bias_result = pd.DataFrame()
	# 计算离散数据的偏度
	for attr in discrete_attrs:
		matching_columns = [col for col in origin_attack_bias.columns if col.startswith(attr)]
		if matching_columns:
			origin_LDP_bias_result[attr] = origin_attack_bias[matching_columns].sum(axis=1) / 2  # 按行计算这些匹配列的和
	for col in [col for col in origin_attack_bias.columns if  # 给出连续数据的偏度
	            not any(col.startswith(attr) for attr in discrete_attrs)]:
		origin_LDP_bias_result[col] = origin_attack_bias[col]
	return origin_LDP_bias_result


def calculate_f2(group):
	"""计算分组的F2分数"""
	# 真实标签：attack_ratio > 0表示被攻击
	y_true = (group['attack_ratio'] > 0).astype(int)
	# 预测标签：detected列（0表示未检测到，1表示检测到）
	y_pred = group['detection'].astype(int)
	
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

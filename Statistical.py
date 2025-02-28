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


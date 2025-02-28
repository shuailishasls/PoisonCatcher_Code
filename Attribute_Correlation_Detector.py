import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import LabelEncoder


# 计算污染程度
def estimate_contamination(n, sigma_j, lambda_c_t, epsilon):
	"""
	根据公式计算污染的估计值。
	n: 样本大小
	sigma_j: 属性j的标准差
	lambda_c_t: 当前时间点的相关性偏差
	epsilon: 隐私保护参数
	"""
	contamination_estimate = np.ceil((n * sigma_j * max(0, lambda_c_t)) / (np.exp(epsilon) - 1))
	return contamination_estimate


def estimate_spearman_corr(df, discrete_attrs , target_features, attacked_mode, mean_ratio, country_attacked_ratio):
	# 对离散属性进行标签编码
	le = LabelEncoder()
	for attr in discrete_attrs:
		if attr in df.columns:
			df[attr] = le.fit_transform(df[attr])
	
	correlation_results = {}
	# 遍历目标特征
	for target_feature in target_features:
		if target_feature in df.columns:
			correlation_results[target_feature] = {}
			# 遍历其他特征
			for other_feature in df.columns:
				if other_feature != target_feature:
					# 检查是否为常量数组
					if len(df[target_feature].unique()) == 1 or len(df[other_feature].unique()) == 1:
						corr, p_value = None, None
					else:
						# 计算 Spearman 相关性
						corr, p_value = spearmanr(df[target_feature], df[other_feature])
					correlation_results[target_feature][other_feature] = {'correlation': abs(corr), 'p_value': p_value}

	data_rows = []
	# 遍历存储相关性结果的字典
	for target_feature, other_features_dict in correlation_results.items():
		for other_feature, result in other_features_dict.items():
			# 构建每一行的数据
			row = {'target_feature': target_feature,
			       'other_feature': other_feature,
			       'correlation': result['correlation'],
			       'p_value': result['p_value'],
			       'attacked_mode': attacked_mode,
			       'real_attacked_ratio': mean_ratio,
			       'expected_attack_ratio': country_attacked_ratio}
			data_rows.append(row)
	
	return pd.DataFrame(data_rows)

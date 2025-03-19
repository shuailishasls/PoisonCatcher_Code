import numpy as np
import pandas as pd


def compute_variance(seq):
	"""
	计算时间序列的样本方差。
	Parameters: seq (array-like): 时间序列数据。
	Returns: float: 样本方差。
	"""
	return np.var(seq, ddof=1)  # ddof=1 gives the sample variance


def compute_max_fluctuation(seq):
	"""
	计算最大波动幅度，即最大值和最小值之间的差值。
	Parameters:	seq (array-like): 时间序列数据。
	Returns: float: 最大波动幅度。
	"""
	return np.max(seq) - np.min(seq)


def compute_autocorrelation(seq):
	"""
	计算时间序列的一阶自相关系数。
	Parameters:	seq (array-like): 时间序列数据。
	Returns: float: 一阶自相关系数。如果分母为0或序列长度不足，则返回0。
	"""
	seq = np.array(seq)
	n = len(seq)
	if n < 2:
		return 0.0
	mean_seq = np.mean(seq)
	numerator = np.sum((seq[:-1] - mean_seq) * (seq[1:] - mean_seq))
	denominator = np.sum((seq - mean_seq) ** 2)
	if denominator == 0:
		return 0.0
	return numerator / denominator


def compute_metrics(seq):
	"""
	计算给定时间序列的稳定性度量：方差、最大波动幅度、一阶自相关系数的绝对值。
	Parameters:	seq (array-like): Time-series data.
	Returns: list: [variance, max fluctuation amplitude, |autocorrelation|]
	"""
	var = compute_variance(seq)
	fluctuation = compute_max_fluctuation(seq)
	ac = compute_autocorrelation(seq)
	return [var, fluctuation, abs(ac)]


def Experiment_3_SQR_Stability_Analysis(result_df_bias):
	"""根据属性的相似性和相关性偏差时间序列计算的稳定性度量来检测属性是否可疑"""
	grouped = result_df_bias.groupby(['attack_ratio', 'attacked_mode', 'attacked_feature'])
	
	# 提取满足条件的组（每个组内 date 不同）
	result_arrays = []
	for name, group in grouped:
		if len(group['date'].unique()) > 1:  # 确保日期不同
			result_arrays.append(group.values)
	
	# 输出结果
	result_df, thresholds_df = pd.DataFrame(), pd.DataFrame()
	for array in result_arrays:
		columns = result_df_bias.columns
		bias_df = pd.DataFrame(columns=columns[:10])
		
		if thresholds_df.empty and array[0, 13] == 0:
			thresholds_df = pd.DataFrame(index=['var', 'fluctuation', 'autocorrelation'], columns=columns[:10])
			for col_idx in range(10):
				metrics_S = compute_metrics(array[:, col_idx])
				thresholds_df.loc['var', columns[col_idx]] = metrics_S[0]
				thresholds_df.loc['fluctuation', columns[col_idx]] = metrics_S[1]
				thresholds_df.loc['autocorrelation', columns[col_idx]] = metrics_S[2]
			continue
		
		if not thresholds_df.empty and array[0, 13] == 0:
			bias_df_result = pd.DataFrame({'detection': 0, 'attacked_mode': [array[0, 10]],
			                               'attacked_feature': [array[0, 12]], 'attack_ratio': [array[0, 13]]})
		
		else:
			for col_idx in range(10):
				metrics_S = compute_metrics(array[:, col_idx])
				bias = 0
				for i in range(3):
					thresholds = thresholds_df.loc[:, columns[col_idx]].to_list()  # 检查两个序列中的三个指标中是否有任何一个超过了各自的阈值。
					if metrics_S[i] > thresholds[i]:
						bias += metrics_S[i] - thresholds[i]
				bias_df.loc[0, columns[col_idx]] = bias
		
			bias_df_result = pd.DataFrame({'detection': 1 if bias_df.idxmax(axis=1).values[0] == array[0, 12] else 0,
			                               'attacked_mode': [array[0, 10]], 'attacked_feature': [array[0, 12]],
			                               'attack_ratio': [array[0, 13]]})
		
		result_df = pd.concat([result_df, bias_df_result], ignore_index=True)
		
	return result_df

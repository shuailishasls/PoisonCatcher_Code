import numpy as np
import pandas as pd
from scipy import stats


# 计算置信区间
def calculate_ci(g):
	if len(g) < 2: return np.nan, np.nan
	ci = stats.t.interval(0.95, len(g) - 1, loc=np.mean(g), scale=stats.sem(g))
	return pd.Series({'ci_low': ci[0], 'ci_high': ci[1]})


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

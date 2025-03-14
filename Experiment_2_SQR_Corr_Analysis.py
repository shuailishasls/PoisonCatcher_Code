from pathlib import Path
import pandas as pd
import os
import numpy as np
from sklearn.cross_decomposition import CCA
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import warnings
from Statistic_In_Experiment import bootstrap_confidence_interval_df

# 忽略 RuntimeWarning
warnings.filterwarnings('ignore', category=RuntimeWarning)


# 核心分析类
class CorrelationAnalyzer:
	def __init__(self, df, discrete_attrs):
		self.df = df.drop(columns='date', errors='ignore')
		self.discrete_attrs = discrete_attrs
		self.cont_cols = [c for c in self.df.columns
		                  if not any(c.startswith(a) for a in self.discrete_attrs)]
	
	def _get_group_cols(self, base_name):
		return [c for c in self.df.columns if c.startswith(f"{base_name}_")]
	
	@staticmethod
	def _remove_last_column(X):
		return X.iloc[:, :-1] if len(X.columns) > 1 else X
	
	# 新增连续变量间分析
	def continuous_vs_continuous(self):
		"""计算连续变量间的Pearson相关系数"""
		corr_matrix = self.df[self.cont_cols].corr().stack().reset_index()
		corr_matrix.columns = ['var1', 'var2', 'correlation']
		return corr_matrix[corr_matrix['var1'] < corr_matrix['var2']]  # 去除重复项
	
	def continuous_vs_group(self, cont_col, disc_base):
		disc_cols = self._get_group_cols(disc_base)
		X = self._remove_last_column(self.df[disc_cols])
		
		pearson_corrs = X.corrwith(self.df[cont_col], method='spearman')
		weighted_mean = self._weighted_correlation(pearson_corrs)
		
		X_ols = add_constant(X)
		model = OLS(self.df[cont_col], X_ols).fit()
		mapped_corr = self._r2_to_corr(model.rsquared, pearson_corrs)
		
		return {
			'analysis_type': 'continuous-group',
			'var1': cont_col,
			'var2': disc_base,
			'correlation': weighted_mean,
			'mapped_r': mapped_corr
		}
	
	def group_vs_group(self, group1, group2):
		cols1 = self._remove_last_column(self.df[self._get_group_cols(group1)])
		cols2 = self._remove_last_column(self.df[self._get_group_cols(group2)])
		
		cca = CCA(n_components=1, max_iter=10000)
		cca.fit(cols1, cols2)
		X_c, Y_c = cca.transform(cols1, cols2)
		cca_corr = np.corrcoef(X_c.T, Y_c.T)[0, 1]
		
		return {
			'analysis_type': 'group-group',
			'var1': group1,
			'var2': group2,
			'correlation': cca_corr
		}
	
	@staticmethod
	def _weighted_correlation(corrs):
		weights = np.abs(corrs)
		return np.sum(corrs * weights) / weights.sum() if weights.sum() != 0 else 0
	
	@staticmethod
	def _r2_to_corr(r_squared, corrs):
		if len(corrs) == 0:
			return 0
		return np.sign(corrs.iloc[np.argmax(np.abs(corrs))]) * np.sqrt(r_squared)


def analyzer_corr(df, discrete_attrs):
	analyzer = CorrelationAnalyzer(df, discrete_attrs)  # 初始化分析器
	
	results = []
	# 1. 连续 vs 连续
	cont_cont = analyzer.continuous_vs_continuous()
	for _, row in cont_cont.iterrows():
		results.append({
			'analysis_type': 'continuous-continuous',
			'var1': row['var1'],
			'var2': row['var2'],
			'correlation': row['correlation']
		})
	# 2. 连续 vs 离散组
	for cont_col in analyzer.cont_cols:
		for disc_base in discrete_attrs:
			if analyzer._get_group_cols(disc_base):
				res = analyzer.continuous_vs_group(cont_col, disc_base)
				results.append({
					'analysis_type': res['analysis_type'],
					'var1': res['var1'],
					'var2': res['var2'],
					'correlation': res['correlation']
				})
	# 3. 离散组 vs 离散组
	for i in range(len(discrete_attrs)):
		for j in range(i + 1, len(discrete_attrs)):
			res = analyzer.group_vs_group(discrete_attrs[i], discrete_attrs[j])
			results.append({
				'analysis_type': res['analysis_type'],
				'var1': res['var1'],
				'var2': res['var2'],
				'correlation': res['correlation']
			})
	return pd.DataFrame(results)


def generate_corr_matrix(true_feature_columns, corr_df, discrete_attrs):
	# 初始化全零矩阵
	matrix = pd.DataFrame(np.zeros((len(true_feature_columns), len(true_feature_columns))),
	                      index=true_feature_columns, columns=true_feature_columns)
	for _, row in corr_df.iterrows():
		var1, var2, corr = row['var1'], row['var2'], row['correlation']
		
		# 处理离散组基名映射
		var1 = next((a for a in discrete_attrs if var1.startswith(a)), var1)
		var2 = next((a for a in discrete_attrs if var2.startswith(a)), var2)
		
		if var1 in true_feature_columns and var2 in true_feature_columns:
			matrix.at[var1, var2] = corr
			matrix.at[var2, var1] = corr  # 确保对称性
	
	# 设置对角线为1
	arr = matrix.values
	np.fill_diagonal(arr, 1)
	
	return pd.DataFrame(arr, index=matrix.index, columns=matrix.columns)


def find_max_diff_row(attacked_matrix, history_clean_ds_matrix):
	"""
	计算两个 DataFrame 对应元素差值的绝对值，排序并找出前 5 个最大差值。
	参数:
	attacked_matrix (pd.DataFrame): 第一个 DataFrame
	history_clean_ds_matrix (pd.DataFrame): 第二个 DataFrame
	返回:
	pd.Series: 包含前 1 个最大差值的 Series，索引为 (index, feature) 元组
	"""
	row_sums = (attacked_matrix - history_clean_ds_matrix).abs().sum(axis=1)  # 按行求和
	sorted_row_sums = row_sums.sort_values(ascending=False)  # 对行和进行排序（降序）
	diff_row_sums = pd.DataFrame([row_sums.values.flatten()], columns=row_sums.index)
	return diff_row_sums, sorted_row_sums


def compute_dis_with_ci(max_diff_row, confidence_intervals):
	# 计算每个值与 lower_bound 和 upper_bound 的距离
	distance_dict = {}
	for col in max_diff_row.columns:
		value = max_diff_row[col].values[0]
		if confidence_intervals.loc[col, 'lower_bound'] <= value <= confidence_intervals.loc[col, 'upper_bound']:
			min_distance = 0
		else:
			lower_distance = abs(value - confidence_intervals.loc[col, 'lower_bound'])
			upper_distance = abs(value - confidence_intervals.loc[col, 'upper_bound'])
			min_distance = min(lower_distance, upper_distance)
		distance_dict[col] = min_distance
	
	# 将结果转换为 DataFrame
	distance_df = pd.DataFrame.from_dict(distance_dict, orient='index', columns=['distance'])
	
	# 从大到小排序
	sorted_distance_df = distance_df.sort_values(by='distance', ascending=False)
	return sorted_distance_df.index[0]


def Experiment_2_SQR_Corr_Analysis(continue_attrs, discrete_attrs, drop_column, clean_ds_sqr, confidence_level):
	result_df, diff_clean_result, diff_clean_ci = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
	
	# 基线数据提取
	df = pd.read_csv('./File/Attacked_Dataset/humidity.csv')
	filtered_rows = df[(df['attack_ratio'] == 0) & (df['attacked_mode'] == 'DIPA')]
	
	history_clean_ds_sqr_corr = analyzer_corr(clean_ds_sqr, discrete_attrs)
	history_clean_ds_matrix = generate_corr_matrix(continue_attrs + discrete_attrs, history_clean_ds_sqr_corr, discrete_attrs)
	
	for _, row in filtered_rows.iterrows():
		current_new_attack_df = pd.DataFrame([row.copy().drop(drop_column)])
		mixed_dataset = pd.concat([clean_ds_sqr, current_new_attack_df], axis=0, ignore_index=True)
		attacked_ds_sqr_corr = analyzer_corr(mixed_dataset, discrete_attrs)
		attacked_matrix = generate_corr_matrix(continue_attrs + discrete_attrs, attacked_ds_sqr_corr,
		                                       discrete_attrs)
		# 找出历史干净数据集和当前被攻击数据集相关性差异
		max_diff_row, _ = find_max_diff_row(attacked_matrix, history_clean_ds_matrix)
		diff_clean_ci = pd.concat([diff_clean_ci, max_diff_row], ignore_index=True)
	
	# 按列计算置信区间——基线
	confidence_intervals = bootstrap_confidence_interval_df(diff_clean_ci, confidence_level)
	
	# -----相关性判断-----
	for i, file in enumerate(set(os.listdir('./File/Attacked_Dataset/'))):
		df = pd.read_csv(os.path.join('./File/Attacked_Dataset/', file))
		df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce').dt.date
		df = df.fillna(0)
		df = df[df['attack_ratio'] != 0]
		attacked_feature = os.path.splitext(os.path.basename(file))[0]
		
		for index, row in df.iterrows():
			# 历史干净数据集SQR
			clean_df_new = df[(df['attack_ratio'] == 0) & (df['attacked_mode'] == 'DIPA') & (df['date'] < row['date'])]
			clean_df_new = pd.concat([clean_ds_sqr, clean_df_new.copy().drop(drop_column, axis=1)],
			                         axis=0, ignore_index=True)
			
			# 当前被攻击数据集SQR
			current_new_attack_df = pd.DataFrame([row.copy().drop(drop_column)])
			
			# 相关性分析
			if not clean_df_new.empty:
				history_clean_ds_sqr_corr = analyzer_corr(clean_df_new, discrete_attrs)
				mixed_dataset = pd.concat([clean_df_new, current_new_attack_df], axis=0, ignore_index=True)

			else:
				history_clean_ds_sqr_corr = analyzer_corr(clean_ds_sqr, discrete_attrs)
				mixed_dataset = pd.concat([clean_ds_sqr, current_new_attack_df], axis=0, ignore_index=True)
			
			# 生成相关性矩阵
			history_clean_ds_matrix = generate_corr_matrix(continue_attrs + discrete_attrs,
			                                               history_clean_ds_sqr_corr, discrete_attrs)
			
			attacked_ds_sqr_corr = analyzer_corr(mixed_dataset, discrete_attrs)
			attacked_matrix = generate_corr_matrix(continue_attrs + discrete_attrs, attacked_ds_sqr_corr, discrete_attrs)

			# 找出历史干净数据集和当前被攻击数据集相关性差异
			max_diff_row, sorted_row = find_max_diff_row(attacked_matrix, history_clean_ds_matrix)
			# if attacked_feature in discrete_attrs:
			success = int(compute_dis_with_ci(max_diff_row, confidence_intervals) == attacked_feature or sorted_row.index[0] == attacked_feature)
			# else:
			# 	success = int(sorted_row.index[0] == attacked_feature)
			max_diff_result = pd.DataFrame({'successful_detection': [success], 'attacked_mode': [row['attacked_mode']],
			                                'attack_ratio': [row['attack_ratio']], 'date': [row['date']],
			                                'attacked_feature': [attacked_feature]})
		
			diff_clean_result = pd.concat([diff_clean_result, max_diff_result], ignore_index=True)

	return diff_clean_result

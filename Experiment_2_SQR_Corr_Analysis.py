from pathlib import Path
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from sklearn.cross_decomposition import CCA
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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
		
		pearson_corrs = X.corrwith(self.df[cont_col])
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


def generate_corr_matrix(true_feature_columns, corr_df):
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
	abs_diff = (attacked_matrix - history_clean_ds_matrix).abs()  # 计算对应元素差值的绝对值
	row_sums = abs_diff.sum(axis=1)  # 按行求和
	sorted_row_sums = row_sums.sort_values(ascending=False)  # 对行和进行排序（降序）
	return sorted_row_sums.head(1)


if __name__ == "__main__":
	continue_attrs = ['temperature_celsius', 'humidity', 'cloud', 'visibility_km', 'uv_index',
	                  'air_quality_Ozone', 'air_quality_Nitrogen_dioxide']  # 连续属性
	discrete_attrs = ['wind_direction', 'air_quality_us-epa-index', 'air_quality_gb-defra-index']  # 离散属性
	drop_column = ['attack_ratio', 'attacked_mode', 'Fault Tolerance']
	
	clean_ds_sqr = pd.read_csv('./File/History_Clean_Dataset_SQR.csv')
	clean_ds_sqr['date'] = pd.to_datetime(clean_ds_sqr['date'], format='%Y-%m-%d', errors='coerce').dt.date
	
	folders = ('./File/Attacked_Dataset/', './File/Attacked_Dataset_Result/')
	file_sets = [set(os.listdir(folder)) for folder in folders]
	unique_files = file_sets[0] - file_sets[1]
	
	for file in tqdm(unique_files):
		experiment_result = []
		
		file_path = os.path.join(folders[0], file)
		df = pd.read_csv(file_path)
		df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce').dt.date
		df = df.fillna(0)
		attacked_feature = os.path.splitext(os.path.basename(file))[0]
		
		for index, row in df.iterrows():
			# 历史干净数据集SQR
			clean_df_new = df[(df['attack_ratio'] == 0) & (df['attacked_mode'] == 'RPVA') & (df['date'] < row['date'])]
			clean_df_new = pd.concat([clean_ds_sqr, clean_df_new.copy().drop(drop_column, axis=1)], axis=0,
			                         ignore_index=True)
			if not clean_df_new.empty:
				history_clean_ds_sqr_corr = analyzer_corr(clean_df_new, discrete_attrs)
			else:
				history_clean_ds_sqr_corr = analyzer_corr(clean_ds_sqr, discrete_attrs)
			history_clean_ds_matrix = generate_corr_matrix(continue_attrs + discrete_attrs, history_clean_ds_sqr_corr)
			
			# 当前被攻击数据集SQR
			current_new_attack_df = pd.DataFrame([row.copy().drop(drop_column)])
			mixed_dataset = pd.concat([clean_df_new, current_new_attack_df], axis=0, ignore_index=True)
			attacked_ds_sqr_corr = analyzer_corr(mixed_dataset, discrete_attrs)
			attacked_matrix = generate_corr_matrix(continue_attrs + discrete_attrs, attacked_ds_sqr_corr)
			
			# 找出历史干净数据集和当前被攻击数据集相关性差异，并按差异大小排序
			max_diff_row = find_max_diff_row(attacked_matrix, history_clean_ds_matrix)
			successful_detection = 1 if max_diff_row.index[0] == attacked_feature else 0
			experiment_result.append([row['attacked_mode'], row['attack_ratio'], row['date'], successful_detection])
		
		pd.DataFrame(experiment_result, columns=['attack_mode', 'attack_ratio', 'attack_time', 'detection']).to_csv(
			f'./File/Attacked_Dataset_Result/{attacked_feature}.csv', index=False)
	
	# 读取所有CSV文件并合并数据
	folders = ['./File/Attacked_Dataset_Result/']
	file_list = [os.path.join(folders[-1], f) for f in os.listdir(folders[-1]) if f.endswith('.csv')]
	
	# 创建画布
	fig = plt.figure(figsize=(24, 18))
	fig.subplots_adjust(top=0.92, hspace=0.4, wspace=0.3)
	plt.rcParams['figure.dpi'] = 1000  # 设置图片清晰度
	
	# 获取所有attack_mode类型
	all_modes = []
	for file in file_list:
		all_modes.extend(pd.read_csv(file)['attack_mode'].unique())
	attack_modes = sorted(list(set(all_modes)))
	
	# 创建颜色和标记映射
	colors = ['red', 'green', 'blue', 'black']
	markers = ['o', 's', '^', 'd']  # 圆形、正方形、三角形、菱形
	style_map = {mode: {'color': colors[i % len(colors)], 'marker': markers[i % len(markers)]}
	             for i, mode in enumerate(attack_modes)}
	
	# 绘制每个子图
	for idx, file in enumerate(file_list, 1):
		ax = fig.add_subplot(4, 5, idx)  # 5行4列布局
		
		df = pd.read_csv(file)
		groups = df['attack_ratio'].diff().gt(0.02).cumsum()
		# 按分组计算 attack_ratio 的均值并替换原数据
		df['attack_ratio'] = round(df.groupby(groups)['attack_ratio'].transform('mean'), 3)
		df_grouped = df.groupby(['attack_mode', 'attack_ratio'])['detection'].mean().reset_index()
		
		for mode in attack_modes:
			mode_data = df_grouped[df_grouped['attack_mode'] == mode]
			if not mode_data.empty:
				ax.plot(mode_data['attack_ratio'], mode_data['detection'],
				        **style_map[mode], linewidth=0.5)
		
		# 子图设置
		title = os.path.splitext(os.path.basename(file))[0]
		ax.set_title(title, fontsize=16, pad=8)
		ax.set_xlabel('Attack Ratio', fontsize=16)
		ax.set_ylabel('Success Rate of Detection', fontsize=16)
		ax.tick_params(axis='both', which='major', labelsize=14)
		ax.set_ylim(-0.1, 1.1)
		ax.grid(True, alpha=0.3)
	
	# 创建统一图例
	legend_elements = [Line2D([0], [0], marker=style_map[mode]['marker'], color=style_map[mode]['color'],
	                          label=mode, markersize=8, linestyle='None') for mode in attack_modes]
	
	fig.legend(handles=legend_elements, loc='upper center', ncol=4, fontsize=16, title='Attack Modes',
	           title_fontsize=16, frameon=True, bbox_to_anchor=(0.5, 1.05))
	
	plt.tight_layout()
	plt.savefig('./File/PDF/Experiment_2_SQR_Corr_Analysis.pdf', format='pdf')
	plt.show()

import pandas as pd
import numpy as np
from scipy import stats


def adjust_ratio(group):
	# 找到基准行（attack_ratio=0）
	base_row = group[group['attack_ratio'] == 0]
	base_mean = base_row['mean_ratio'].iloc[0]
	
	# 调整非零行的mean_ratio
	adjusted_group = group.copy()
	adjusted_group.loc[adjusted_group['attack_ratio'] != 0, 'mean_ratio'] -= base_mean
	adjusted_group.loc[adjusted_group['mean_ratio'] < 0, 'mean_ratio'] = 0
	
	return adjusted_group[adjusted_group['attack_ratio'] != 0]


def Experiment_4_Ratio_Estimates(origin_df, bias_df):
	discrete_attrs = {'air_quality_us-epa-index': 6, 'air_quality_gb-defra-index': 10}  # 离散属性
	results, std_dev_dict = [], {}
	
	selected_columns = origin_df.columns[2:]
	for col in selected_columns:  # 计算每列的标准差
		std_dev = origin_df[col].std()
		std_dev_dict[col] = std_dev
	
	grouped = bias_df.groupby(['attack_ratio', 'attacked_feature', 'attacked_mode'])
	for name, group in grouped:
		attack_ratio, attacked_feature, attacked_mode = name
		num_unique_dates = len(sorted(group['date'].unique()))
		
		exp1_dates = group[:num_unique_dates]
		exp2_dates = group[num_unique_dates:]
		
		for exp_name, datas in [('exp1', exp1_dates), ('exp2', exp2_dates)]:
			deviation_values = datas[attacked_feature]
			ratio_values = []
			if exp_name == 'exp1':
				for i in deviation_values:
					if name[1] not in discrete_attrs:
						ratio_values.append(i / (2 * (np.e - 1)))
					else:
						ratio_values.append(i / (discrete_attrs[name[1]] * (np.e - 1)))
			
			else:
				for i in deviation_values:
					ratio_values.append((i * std_dev_dict[name[1]]) / (np.e - 1))
			
			results.append({
				'mean_ratio': np.mean(ratio_values),
				'attack_ratio': attack_ratio,
				'attacked_feature': attacked_feature,
				'attacked_mode': attacked_mode,
				'experiment': exp_name
			})
	
	output_df = pd.DataFrame(results)
	
	output_df = output_df.groupby(['attacked_feature', 'attacked_mode', 'experiment']).apply(adjust_ratio).reset_index(
		drop=True)
	
	# 按照 attacked_feature、attacked_mode、attack_ratio 分组，对 mean_ratio 求和
	grouped_df = output_df.groupby(['attacked_feature', 'attacked_mode', 'attack_ratio'])['mean_ratio'].sum().reset_index()
	filtered_df = grouped_df[grouped_df['attack_ratio'] == 0.056]
	
	# 分离出在 discrete_attrs 中的 attacked_feature 和不在其中的
	in_discrete = filtered_df[filtered_df['attacked_feature'].isin(discrete_attrs.keys())]
	not_in_discrete = filtered_df[~filtered_df['attacked_feature'].isin(discrete_attrs.keys())]
	
	# 只选择数值列进行分组求平均
	numeric_columns = not_in_discrete[not_in_discrete['mean_ratio'] != 0].select_dtypes(include=['number']).columns
	not_in_discrete_avg = not_in_discrete[not_in_discrete['mean_ratio'] != 0].groupby('attacked_mode')[
		numeric_columns].mean()
	
	# 对于在 discrete_attrs 中的，选择数值列并按 attacked_mode 分组求平均
	in_discrete_avg = in_discrete[in_discrete['mean_ratio'] != 0].groupby('attacked_mode')[numeric_columns].mean()
	
	print('Continuous attribute\n', not_in_discrete_avg)
	print('Discrete attribute\n', in_discrete_avg)

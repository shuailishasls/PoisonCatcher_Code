import os
import numpy as np
import pandas as pd
import Statistical as STATI
import Attack_Simulation_Four as ASF
import Real_Data_Process as Real_Data_Process


def read_files_in_folder(folder_path):
	# 按年读取文件夹的内容存成一个dataframe
	dataframes_dict = {}
	for root, dirs, files in os.walk(folder_path):
		for file in files:
			file_path = os.path.join(root, file)
			try:
				data = pd.read_csv(file_path)
				dataframes_dict[file.split('.')[0]] = data
			except Exception as e:
				print(f"Error reading file {file_path}: {e}")
	return dataframes_dict


def process_date_list(date_list, threshold):
	# 按时间顺序对日期列表进行排序
	sorted_dates = sorted(date_list)
	# 计算分割点的索引
	split_index = int(len(sorted_dates) * threshold)
	# 分割数据集为前后两部分
	part1 = sorted_dates[:split_index]
	part2 = sorted_dates[split_index:]
	return part1, part2


def flatten_dict(data, discrete_attr):
	# 将输入的字典数据进行扁平化处理，转换为 DataFrame 格式
	data = pd.DataFrame(data).T
	data['date'] = data.index
	data = data.reset_index(drop=True)
	cols = ['date'] + [col for col in data.columns if col != 'date']
	data = data[cols]
	data['date'] = pd.to_datetime(data['date']).dt.date
	
	new_columns = []
	for _, row in data.iterrows():
		new_row = {}
		for col in data.columns:
			if col in discrete_attr:
				value = row[col]
				if isinstance(value, dict):
					for key1, val in value.items():
						new_col_name = f"{col}_{key1}"
						new_row[new_col_name] = val
			else:
				# 直接将非 discrete_attrs 中的列值添加到新行
				new_row[col] = row[col]
		new_columns.append(new_row)
	result_df = pd.DataFrame(new_columns).fillna(0)
	
	return result_df


def generate_attacked_df(epsilon, discrete_attrs, attack_countries, selected_attacked_features, attacked_modes,
                         country_attacked_ratio, current_ds, dfs_LDP_dict, dfs_dict, alpha_df, true_attack_ratio,
                         processed_data, processed_data_non_LDP):
	"""
	根据原始数据集生成攻击数据集
	:param confidence_level: 置信概率
	:param epsilon: 隐私预算
	"""
	# ------------------------------------开始攻击------------------------------------
	for feature in selected_attacked_features:  # 确定攻击特征
		new_LDP_sqr, attack_alpha = pd.DataFrame(), pd.DataFrame()  # | 攻击理论阈值
		for attacked_mode in attacked_modes:  # 确定攻击模式
			for i in range(len(country_attacked_ratio)):  # 确定攻击程度
				real_attack_ratio_list, temp_LDP_sqr = [], pd.DataFrame()
				for date_obj in current_ds:  # 确定攻击时间
					current_df = dfs_LDP_dict[date_obj.strftime("%Y-%m-%d")].copy()
					ture_df = dfs_dict[date_obj.strftime("%Y-%m-%d")].copy()
					wait_attack_ds = current_df.copy()  # 被LDP处理后的数据
					wait_attack_ds['attacked'] = 0
					
					for user in attack_countries[i]:  # 定位被攻击的国家
						origin_data = wait_attack_ds.loc[wait_attack_ds['country'] == user, feature].to_list()
						
						if feature not in discrete_attrs:
							if attacked_mode == 'ROPA':  # 输出攻击
								attacked_value = ASF.ROPA('Laplace', [current_df.loc[:, feature].min(),
								                                      current_df.loc[:, feature].max()], origin_data)
							
							elif attacked_mode == 'DIPA':  # 输入攻击
								ture_df_copy = ture_df.loc[processed_data['country'] == user, feature].to_list()
								ture_domain = processed_data_non_LDP.loc[processed_data_non_LDP['country'] == user, feature]
								attacked_value = ASF.DIPA('Laplace', [min(ture_domain), max(ture_domain)],
								                          origin_data, epsilon, ture_df_copy)
							
							elif attacked_mode == 'DPPA':  # 规则攻击
								attacked_value = ASF.DPPA(
									ture_df.loc[wait_attack_ds['country'] == user, feature].to_list(),
									origin_data, 'Laplace')
							
							else:  # 最大增益攻击
								attacked_value = ASF.SMGPA(STATI.laplace_mean_estimation(processed_data[feature]),
								                         'Laplace', origin_data, epsilon=epsilon)
						
						else:
							domain = processed_data[feature].value_counts().sort_values(ascending=False).index.tolist()
							
							if attacked_mode == 'ROPA':  # 输出攻击
								attacked_value = ASF.ROPA('GRR', domain, origin_data)
							
							elif attacked_mode == 'DIPA':  # 输入攻击
								ture_df_copy = ture_df.loc[processed_data['country'] == user, feature].copy()
								attacked_value = ASF.DIPA('GRR', domain, origin_data, epsilon, ture_df_copy)
							
							elif attacked_mode == 'DPPA':  # 规则攻击
								true_value = ture_df.loc[wait_attack_ds['country'] == user, feature].to_list()
								attacked_value = ASF.DPPA(true_value, origin_data, 'GRR', domain)
							
							else:  # 最大增益攻击
								target_item = STATI.grr_frequency_estimation(processed_data[feature], domain, epsilon)
								attacked_value = ASF.SMGPA(max(target_item, key=target_item.get), 'GRR',
								                           origin_data, domain, epsilon)
						
						wait_attack_ds.loc[wait_attack_ds['country'] == user, feature] = attacked_value
						wait_attack_ds.loc[wait_attack_ds['country'] == user, 'attacked'] = 1
					
					# ------------------------------------判断数据集是否被攻击------------------------------------
					# 对当前时刻攻击数据集中每个属性计算统计结果
					current_attacked_sqr = pd.DataFrame(
						[{f"{k}_{ik}" if k in discrete_attrs else k: v if k not in discrete_attrs else iv for k, v in
						  STATI.calculate_features_statistics(wait_attack_ds.iloc[:, 2:-1], discrete_attrs,
						                                      epsilon).items()
						  for ik, iv in (v.items() if k in discrete_attrs else [(None, v)])}])
					
					current_attacked_sqr['date'] = date_obj
					current_attacked_sqr['Fault Tolerance'] = alpha_df[feature].values[0]
					temp_LDP_sqr = pd.concat([temp_LDP_sqr, current_attacked_sqr], ignore_index=True)
				
				temp_LDP_sqr['attack_ratio'] = true_attack_ratio[i]
				temp_LDP_sqr['attacked_mode'] = attacked_mode
				new_LDP_sqr = pd.concat([new_LDP_sqr, temp_LDP_sqr], ignore_index=True)
		
		new_LDP_sqr.to_csv(f'./File/Attacked_Dataset/{str(feature)}.csv', index=False)
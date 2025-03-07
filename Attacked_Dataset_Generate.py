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


def generate_attacked_df(exp, times, date_threshold, confidence_level, epsilon, discrete_attrs, continue_attrs):
	"""
	根据原始数据集生成攻击数据集
	:param exp: 生成实验几的结果
	:param times: 时间实例的宽度(可选1、3、9、31)
	:param date_threshold: 被攻击时间占比
	:param confidence_level: 置信概率
	:param epsilon: 隐私预算
	"""
	country_attacked_ratio = (np.arange(0, 52, 3).astype(float) / 100).tolist()  # 被攻击的国家比例
	attacked_modes = ['DIPA', 'DPPA', 'ROPA']  # 攻击者的攻击手段
	attack_countries, sqr_accuracy, LDP_sqr, origin_sqr, attack_sqr = [], pd.DataFrame(), {}, {}, {}
	
	if exp == 1 or exp == 2:
		selected_attacked_features = continue_attrs + discrete_attrs
	else:
		selected_attacked_features = continue_attrs + discrete_attrs
	
	print(f'数据预处理ing...')
	Real_Data_Process.process_csv(discrete_attrs, './File/GlobalWeatherRepository.csv', epsilon, times,
	                              selected_attacked_features)
	processed_data = pd.read_csv("./File/Preprocessing_Data.csv")  # 读取预处理数据
	processed_data_non_LDP = pd.read_csv("./File/Preprocessing_Data(non_LDP).csv")  # 读取预处理数据
	dfs_LDP_dict = read_files_in_folder('./File/Divide_data_by_time')  # 按日读取文件夹的内容存成一个dataframe
	dfs_dict = read_files_in_folder('./File/Divide_data_by_time(non_LDP)')
	
	# 确定攻击dates
	group_by_date = pd.to_datetime(processed_data['date'].unique()).date
	history_ds, current_ds = process_date_list(group_by_date, date_threshold)
	
	# 确定被攻击的countries
	np.random.seed(2)
	common_countries = list(processed_data['country'].unique())
	for ratio in country_attacked_ratio:
		size = int(len(common_countries) * ratio)
		if attack_countries:
			new = np.random.choice([x for x in common_countries if x not in attack_countries[-1]],
			                       size - len(attack_countries[-1]), replace=False)
			attack_countries.append(np.sort(np.concatenate([attack_countries[-1], new])))
		else:
			attack_countries.append(np.sort(np.random.choice(common_countries, size, replace=False)))
	
	# 计算实际攻击比例
	valid_dfs = [dfs_LDP_dict.get(d.strftime('%Y-%m-%d')) for d in current_ds if d.strftime('%Y-%m-%d') in dfs_LDP_dict]
	average_rows = sum(len(df) for df in valid_dfs) / len(valid_dfs) if valid_dfs else 0
	true_attack_ratio = [round((len(sub_list) / average_rows) * times, 3) if average_rows else 0 for sub_list in
	                     attack_countries]
	
	# 对扰动数据集中每个属性计算统计结果
	for key, df in dfs_LDP_dict.items():
		LDP_sqr[key] = STATI.calculate_features_statistics(df.iloc[:, 2:], discrete_attrs, epsilon)
	LDP_sqr = flatten_dict(LDP_sqr, discrete_attrs)  # 将输入的3维字典2维化处理
	LDP_sqr[LDP_sqr['date'].isin(history_ds)].to_csv(f'./File/History_Clean_LDP_SQR.csv', index=False)
	
	# 对原始数据集中的每个属性计算统计结果
	for key, df in dfs_dict.items():
		origin_sqr[key] = STATI.calculate_features_statistics(df.iloc[:, 2:], discrete_attrs, epsilon)
	origin_sqr = flatten_dict(origin_sqr, discrete_attrs)  # 将输入的3维字典2维化处理
	origin_sqr.to_csv(f'./File/History_Clean_origin_SQR.csv', index=False)
	
	# 计算原始数据与LDP处理后的数据的偏差
	origin_LDP_bias_result = STATI.compute_bias(discrete_attrs, (LDP_sqr.iloc[:, 1:] - origin_sqr.iloc[:, 1:]).abs())
	
	# 计算基线数据集中每个属性的容错阈值
	alpha_df = STATI.calculate_ft(origin_sqr.iloc[:, 1:], len(dfs_dict[next(iter(dfs_dict))]), epsilon,
	                              confidence_level,
	                              discrete_attrs).T
	alpha_df.to_csv(f'./File/alpha_df.csv', index=False)
	# 判断偏差是否在阈值范围内
	bias_out_FT_df = origin_LDP_bias_result.copy()
	for col in origin_LDP_bias_result.columns:
		bias_out_FT_df[col] = (origin_LDP_bias_result[col] > alpha_df[col].values[0]).astype(int)
		if bias_out_FT_df[col].sum() != 0:
			print(col)
	
	# ------------------------------------开始攻击------------------------------------
	for feature in selected_attacked_features:  # 确定攻击特征
		new_LDP_sqr, attack_alpha = pd.DataFrame(), pd.DataFrame()  # | 攻击理论阈值
		for attacked_mode in attacked_modes:  # 确定攻击模式
			for i in range(len(country_attacked_ratio)):  # 确定攻击程度
				real_attack_ratio_list, temp_LDP_sqr = [], pd.DataFrame()
				for date_obj in current_ds:  # 确定攻击时间
					current_df = dfs_LDP_dict[date_obj.strftime("%Y-%m-%d")]
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
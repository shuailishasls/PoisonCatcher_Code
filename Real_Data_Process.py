#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：PoisonCatcher
@File ：Real_Data_Process.py
@Author ：SLS
@Date ：25.01.21
"""
import os
import random
import hashlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import datetime
from sklearn.preprocessing import QuantileTransformer


def group_data(df, time_instances):
	# 生成分组标识
	dates = sorted(df['date'].unique())
	groups = {g[0]: g for i in range(0, len(dates), time_instances) if
	          len(g := dates[i:i + time_instances]) == time_instances}
	
	# 构建分组DataFrame
	dfs = []
	for start_date, date_group in groups.items():
		group_df = (
			df[df['date'].isin(date_group)].sort_values('date').assign(group_start=start_date).reset_index(drop=True))
		dfs.append(group_df)
	
	# 合并并保持原始列顺序
	result_df = pd.concat(dfs, ignore_index=True)[['group_start'] + df.columns.tolist()]
	result_df = result_df.drop(columns=["date"])
	return result_df.rename(columns={'group_start': 'date'})


def empty_folder_files(folder_path):
	if os.path.exists(folder_path):
		for file in (f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))):
			try:
				os.remove(os.path.join(folder_path, file))
			except Exception as e:
				print(f"删除文件 {file} 出错: {e}")
	else:
		print(f"文件夹 {folder_path} 不存在。")


def process_csv(disc_att, input_file, epsilon, time_instances, selected_attacked_features):
	"""
	读取原始 CSV 文件并对其进行预处理
	参数:
	disc_att (str): 离散属性的列名
	input_file (str): 输入 CSV 文件的路径
	"""
	df = pd.read_csv(input_file)
	empty_folder_files(f'./File/Divide_data_by_time(non_LDP)/')
	empty_folder_files(f'./File/Divide_data_by_time/')
	
	# 删除指定列，删除原因：重复
	df = df.drop(columns=["location_name", "latitude", "longitude", "timezone", "last_updated_epoch", "wind_mph",
	                      "temperature_fahrenheit", "wind_degree", "pressure_in", "precip_in", "feels_like_fahrenheit",
	                      "visibility_miles", "gust_mph", "sunrise", "sunset", "moonrise", "moonset", "moon_phase",
	                      "moon_illumination"])
	
	# 将 last_updated 列的值转换为 datetime.date 形式，并改名为 ’date‘
	df['last_updated'] = pd.to_datetime(df['last_updated']).dt.date
	df = df.rename(columns={'last_updated': 'date'})
	df = df[['date', 'country'] + selected_attacked_features]

	# 筛选出只包含频繁出现日期的数据行
	date_counts = df['date'].value_counts()
	df = df[df['date'].isin(date_counts[date_counts >= 30].index)]  # 将频繁出现定义为30
	
	# 找出每个 date 中 country 的唯一值
	unique_countries_per_date = df.groupby('date')['country'].unique()
	all_dates = df['date'].unique()
	common_countries = set(unique_countries_per_date[all_dates[0]])
	for date in all_dates[1:]:
		common_countries = common_countries.intersection(set(unique_countries_per_date[date]))
	
	# 按日期和国家去重，确保每个国家在每个日期下只出现一次
	df_filtered = df[df['country'].isin(common_countries)]
	df_filtered = df_filtered.drop_duplicates(subset=['date', 'country'])
	
	# 对不在 disc_att 列表中的列进行[-1,1]标准化处理
	numerical_columns = [col for col in df_filtered.columns if col not in disc_att][2:]
	df_filtered[numerical_columns] = df_filtered[numerical_columns].astype(float)

	qt = QuantileTransformer(output_distribution='normal', random_state=0)
	df_filtered.loc[:, numerical_columns] = qt.fit_transform(df_filtered[numerical_columns])
	
	scaler = MinMaxScaler(feature_range=(-1, 1))
	df_filtered.loc[:, numerical_columns] = scaler.fit_transform(df_filtered[numerical_columns])
	print('总日期数：', len(df_filtered['date'].value_counts()))
	
	# 将日期分割成所需的范围
	df_filtered = group_data(df_filtered, time_instances)
	df_filtered.to_csv('./File/Preprocessing_Data(non_LDP).csv', index=False)
	# 将每个唯一的日期，数据保存为 CSV 文件
	for date in df_filtered['date'].unique():
		df_filtered[df_filtered['date'] == date].to_csv(f'./File/Divide_data_by_time(non_LDP)/{date}.csv', index=False)
	
	df_filtered.to_csv('./File/Preprocessing_Data(non_LDP).csv', index=False)
	
	# 数据LDP处理
	for col in df_filtered.columns[2:]:
		if col not in disc_att:  # 连续属性数据，应用 laplace_mechanism 进行LDP处理
			# 对不在 disc_att 列表中的列
			df_filtered[col] = df_filtered[col].apply(lambda x: laplace_mechanism(x, epsilon))
		else:
			# 若在 disc_att 列表中，对该列应用 grr_mechanism 函数
			df_filtered[col] = df_filtered[col].apply(lambda x: grr_mechanism(x, list(set(df_filtered[col])), epsilon))
	
	df_filtered.to_csv('./File/Preprocessing_Data.csv', index=False)
	
	# 遍历每个唯一的日期
	for date in df_filtered['date'].unique():
		# 筛选出该日期对应的数据
		# 将筛选后的数据保存为 CSV 文件
		df_filtered[df_filtered['date'] == date].to_csv(f'./File/Divide_data_by_time/{date}.csv', index=False)
	
	print(f'数据预处理完成')


def laplace_mechanism(x, epsilon, sensitivity=2):
	"""
	使用Laplace机制对数据进行LDP处理
	参数:
	x - 原始数据
	epsilon - 隐私预算
	sensitivity - 数据敏感度
	返回:
	差分隐私处理后的数据
	"""
	return x + np.random.laplace(loc=0, scale=sensitivity / epsilon)


def grr_mechanism(value, domain, epsilon):
	"""
	使用 General Random Response (GRR) 机制对字符串数据进行本地差分隐私处理
	:param value: 原始的字符串数据
	:param domain: 数据的所有可能取值的集合
	:param epsilon: 隐私预算
	:return: 经过差分隐私处理后的数据
	"""
	# 以概率 p 真实报告原始值
	if random.random() < np.exp(epsilon) / (np.exp(epsilon) + len(domain) - 1):
		return value
	else:
		# 以概率 1 - p 随机选择一个其他的值进行报告
		other_values = [v for v in domain if v != value]
		return random.choice(other_values)

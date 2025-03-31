import os
import random
import pandas as pd
import numpy as np


def process_csv_files(folder_path, output_file, continue_attrs):
	# 遍历文件夹下的所有文件
	for index, filename in enumerate(os.listdir(folder_path)):
		new_rows = pd.DataFrame()
		
		file_path = os.path.join(folder_path, filename)
		df = pd.read_csv(file_path)
		attack_ratio = df['attack_ratio'].unique()[1:]
		attacked_feature = filename.split('.')[0]
		
		if attacked_feature in continue_attrs:
			df = df[[attacked_feature, 'date', 'attack_ratio', 'attacked_mode']]
		else:
			selected_df = df.filter(regex=f'^{attacked_feature}').columns.tolist()
			df = df[selected_df + ['date', 'attack_ratio', 'attacked_mode']]

		grouped = df.groupby(['date', 'attacked_mode'])
		
		# 定义筛选函数
		def filter_rows(group, attack_ratio):
			condition_1 = group['attack_ratio'] == 0
			condition_2 = group['attack_ratio'] == random.choice(attack_ratio)
			return group[condition_1 | condition_2]
		
		filtered_groups = grouped.apply(filter_rows, attack_ratio)  # 对每个分组应用筛选函数
		filtered_df = filtered_groups.reset_index(drop=True)  # 重置索引
		
		# 遍历每个 date 分组
		for _, group in filtered_df.groupby('date'):
			# 筛选出 attack_ratio 为 0 的行
			zero_ratio_rows = group[group['attack_ratio'] == 0]

			first_zero_row = zero_ratio_rows.head(1)  # 取第一行并确保是 DataFrame 类型
			new_rows = pd.concat([new_rows, first_zero_row], ignore_index=True)
			
			# 筛选出 attack_ratio 不为 0 的行
			non_zero_ratio_rows = group[group['attack_ratio'] != 0]
			new_rows = pd.concat([new_rows, non_zero_ratio_rows], ignore_index=True)
		
		new_rows.to_csv(output_file+filename, index=False)
		new_rows.loc[new_rows['attack_ratio'] != 0, 'attack_ratio'] = np.nan
		new_rows.to_csv(output_file + 'test/' + filename, index=False)


if __name__ == "__main__":
	folder_path = './File/Attacked_Dataset/'
	output_file = './File/Human_Expert_Review_Scale/'
	
	continue_attrs = ['temperature_celsius', 'pressure_mb', 'feels_like_celsius', 'gust_kph', 'humidity',
	                  'air_quality_Carbon_Monoxide', 'air_quality_Ozone', 'air_quality_PM10']  # 连续属性
	
	process_csv_files(folder_path, output_file, continue_attrs)

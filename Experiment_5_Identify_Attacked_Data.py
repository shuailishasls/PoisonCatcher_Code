import os
import numpy as np
import pandas as pd
from datetime import date
from sklearn.metrics import fbeta_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def calculate_features(original, attacked, participants, sampling_rate=0.1, sampling_time=200):
	"""	为每个样本计算9个统计特征。	"""
	result = attacked.copy()
	result.iloc[:, 1:-1] = 0
	
	sample_size = int(len(participants) * sampling_rate)
	# 随机抽样
	for _ in range(sampling_time):
		# 采样
		sampled_data = participants[np.random.choice(len(participants), sample_size, replace=False)]
		
		sampled_original = original[original['country'].isin(sampled_data)].iloc[:, 1:-1]
		sampled_attacked = attacked[attacked['country'].isin(sampled_data)].iloc[:, 1:-1]
		
		# 存储每个特征的最终结果
		final_features = []
		
		# 遍历 sampled_original 的每一列
		for col_attacked in sampled_attacked.columns:
			attacked_col = sampled_attacked[col_attacked]
			col_features = []
			
			# 遍历 sampled_attacked 的每一列
			for col_original in sampled_original.columns:
				original_col = sampled_original[col_original]
				
				# 特征1-3：平均值、中位数、方差
				# mean_diff = np.abs(original_col.mean() - attacked_col.mean())
				# median_diff = np.abs(original_col.median() - attacked_col.median())
				# var_diff = np.abs(original_col.var() - attacked_col.var())
				
				# 特征4-5:MAE，KL散度
				# mae = np.mean(np.abs(original_col - attacked_col))
				# kl_divergence = np.sum(
				# 	original_col * np.log(np.maximum(original_col / (attacked_col + 1e-9), 1e-9)))
				
				# 特征6-7：查询偏差和假设检验偏差
				# query_bias = np.abs(original_col.mean() - attacked_col.mean())
				# original_col_reset = original_col.reset_index(drop=True)
				# attacked_col_reset = attacked_col.reset_index(drop=True)
				# hypothesis_bias = np.sum(original_col_reset > attacked_col_reset) / len(original_col_reset)
				
				# 特征8：个体标准偏差
				# std_dev = np.abs(original_col.std() - attacked_col.std())
				
				# 特征9：最大波动
				max_fluctuation = np.max(np.abs(original_col - attacked_col))
				
				# 存储当前列对的特征
				# col_features.append([mean_diff, median_diff, var_diff, mae, kl_divergence, query_bias, hypothesis_bias,
				#                      std_dev, max_fluctuation])
				col_features.append([max_fluctuation])
			
			# 对每个特征求平均
			col_features = np.array(col_features).mean()
			final_features.append(col_features)
		
		# 按列将 final_features 值赋予 result.iloc[selected_indices, 1:-1] 行每个元素
		result.loc[result['country'].isin(sampled_data), result.columns[1:-1]] += final_features

	return result


def train_model(df):
	"""	训练一个随机森林分类器来识别被攻击的item	"""
	# 对 date 和 country 进行编码
	label_encoders = {}
	le = LabelEncoder()
	df['country'] = le.fit_transform(df['country'])
	label_encoders['country'] = le
	
	# 准备特征和标签
	X = df.drop('attacked', axis=1)
	y = df['attacked']
	
	# 划分训练集和测试集
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	
	# 构建随机森林分类器模型并训练
	rf_model = RandomForestClassifier(random_state=42)
	rf_model.fit(X_train, y_train)
	
	# 在测试集上进行预测
	rf_y_pred = rf_model.predict(X_test)
	
	# 计算 F2 分数
	f2_score = fbeta_score(y_test, rf_y_pred, beta=2, average='weighted')
	
	recall = recall_score(y_test, rf_y_pred, average='weighted')
	tn = np.sum((y_test == 0) & (rf_y_pred == 0))
	fp = np.sum((y_test == 0) & (rf_y_pred == 1))
	specificity = tn / (tn + fp)
	g_mean = np.sqrt(recall * specificity)  # 计算 G-mean
	
	return f2_score, g_mean


def process_csv_files(folder_path_in, folder_path_out):
	# 用于存储按攻击模式和攻击程度分组的数据
	grouped_data = {}
	
	# 遍历文件夹中的所有 CSV 文件
	for root, dirs, files in os.walk(folder_path_in):
		for file in files:
			# 从文件名中提取信息
			feature_name, time_info, attack_mode, attack_degree = file.replace('.csv', '').split('+')
			df = pd.read_csv(os.path.join(root, file))
			
			# 构建分组的键
			key = (feature_name, attack_mode, attack_degree)
			
			if key not in grouped_data:
				grouped_data[key] = {
					'country': df['country'],
					'attacked': df['attacked'],
					'time_data': {}
				}
			
			# 提取第二列数据，将列名改为时间
			grouped_data[key]['time_data'][time_info] = df.iloc[:, 1]

	for (feature_name, attack_mode, attack_degree), data in grouped_data.items():
		new_df = pd.DataFrame(data['country'])
		time_df = pd.DataFrame(data['time_data'])
		new_df = pd.concat([new_df, time_df], axis=1)
		new_df['attacked'] = data['attacked']
		new_filename = f"{feature_name}+{attack_mode}+{attack_degree}.csv"
		new_df.to_csv(os.path.join(folder_path_out, new_filename), index=False)


def process_clean_files(history_ds, folder_path, folder_path_out):
	# 用于存储每个特征的数据
	feature_data = {}
	
	# 遍历文件夹中的所有文件
	for filename in os.listdir(folder_path):
		# 尝试将文件名转换为日期格式
		file_date_str = filename.replace('.csv', '')
		file_date = date.fromisoformat(file_date_str)
		if file_date in history_ds:
			file_path = os.path.join(folder_path, filename)
			df = pd.read_csv(file_path)
			
			# 从第二列开始遍历所有特征列
			for col_idx in range(2, len(df.columns)):
				feature_column = df.columns[col_idx]
				
				if feature_column not in feature_data:
					feature_data[feature_column] = df[['country']].copy()
				
				# 提取特征列并改名为日期
				feature_data[feature_column].loc[:, file_date_str] = df[feature_column].values
	
	for df in feature_data.values():
		df['attacked'] = 0
	
	return feature_data


def Identify_Attacked_Data(folder_path, history_ds):
	sampling_rate = 0.2
	results = []
	folder_path_out = './File/Attacked_Dataset_User_Time/'
	
	# process_csv_files(folder_path, folder_path_out)
	
	feature_data = process_clean_files(history_ds, './File/Divide_data_by_time', folder_path_out)
	for feature, data in feature_data.items():
		for filename in os.listdir(folder_path_out):
			attacked_feature, attack_mode, attack_ratio = filename.replace('.csv', '').split('+')
			file_path = os.path.join(folder_path_out, filename)
			if attacked_feature == feature and float(attack_ratio) != 0:
				# 构建特征矩阵
				df = calculate_features(data, pd.read_csv(file_path), data['country'].unique(), sampling_rate)
				f2_score, g_mean = train_model(df)
				results.append([attacked_feature, attack_mode, attack_ratio, f2_score, g_mean])

	result_df = pd.DataFrame(results, columns=['attacked_feature', 'attacked_mode', 'attack_ratio', 'F2 score', 'g_mean'])
	
	return result_df


def Identify_Attacked_Data_normal(folder_path):
	results = []
	# 遍历文件夹中的所有 CSV 文件
	for root, dirs, files in os.walk(folder_path):
		for file in files:
			if file.endswith('.csv'):
				file_path = os.path.join(root, file)
				
				file_name = os.path.basename(file_path)
				attacked_feature, attack_mode, attack_ratio = file_name.replace('.csv', '').split('+')
				f2_score, g_mean = train_model(pd.read_csv(file_path))
				results.append([attacked_feature, attack_mode, attack_ratio, f2_score, g_mean])
	
	# 创建结果 DataFrame
	result_df = pd.DataFrame(results, columns=['attacked_feature', 'attacked_mode', 'attack_ratio', 'F2 score', 'g_mean'])
	
	return result_df
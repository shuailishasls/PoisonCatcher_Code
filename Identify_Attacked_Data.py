import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# Step 1: Load data
def load_data(original_file, attacked_file):
	original_data = pd.read_csv(original_file)
	attacked_data = pd.read_csv(attacked_file)
	return original_data, attacked_data


# Step 2: Calculate statistical features
def calculate_features(original, attacked, sampling_rate=0.1):
	"""
	为每个样本计算9个统计特征。
	"""
	# 采样
	sample_size = int(len(original) * sampling_rate)
	sampled_original = original.sample(sample_size, replace=True)
	sampled_attacked = attacked.sample(sample_size, replace=True)
	
	# 特征1-3：平均值、中位数、方差
	mean_diff = np.abs(sampled_original.mean() - sampled_attacked.mean())
	median_diff = np.abs(sampled_original.median() - sampled_attacked.median())
	var_diff = np.abs(sampled_original.var() - sampled_attacked.var())
	
	# 特征4-5:MAE，KL散度
	mae = np.mean(np.abs(sampled_original - sampled_attacked))
	kl_divergence = np.sum(sampled_original * np.log(sampled_original / (sampled_attacked + 1e-9)))
	
	# 特征6-7：查询偏差和假设检验偏差
	query_bias = np.abs(sampled_original.mean() - sampled_attacked.mean())
	hypothesis_bias = np.sum(sampled_original > sampled_attacked) / len(sampled_original)
	
	# 特征8：个体标准偏差
	std_dev = sampled_original.std()
	
	# 特征9：最大波动
	max_fluctuation = np.max(np.abs(sampled_original - sampled_attacked))
	
	# 组合特征
	features = [
		mean_diff, median_diff, var_diff, mae, kl_divergence,
		query_bias, hypothesis_bias, std_dev.mean(), max_fluctuation.mean()
	]
	return features


# 步骤3：为多个参与者和时间实例构建特征矩阵
def construct_feature_matrix(original_data, attacked_data, participants, time_instances, sampling_rate=0.1):
	"""
	在多个时间实例中为所有参与者构建特征矩阵。
	"""
	feature_matrix = []
	for t in range(time_instances):
		time_features = []
		for participant in range(participants):
			features = calculate_features(
				original_data.iloc[:, participant],
				attacked_data.iloc[:, participant],
				sampling_rate
			)
			time_features.append(features)
		feature_matrix.append(time_features)
	return np.array(feature_matrix)


# 步骤4：训练机器学习模型
def train_model(features, labels):
	"""
	训练一个随机森林分类器来识别被攻击的item
	"""
	X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
	model = RandomForestClassifier(n_estimators=100, random_state=42)
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	print(classification_report(y_test, y_pred))
	return model


# Example Usage
if __name__ == "__main__":
	# Load original and attacked data
	original_file = "generated_original_data.csv"
	attacked_file = "generated_attacked_data.csv"
	original_data, attacked_data = load_data(original_file, attacked_file)
	
	# Parameters
	num_participants = original_data.shape[1]
	num_time_instances = 5
	sampling_rate = 0.1  # Adjust sampling rate if needed
	
	# Construct feature matrix
	features = construct_feature_matrix(original_data, attacked_data, num_participants, num_time_instances,
	                                    sampling_rate)
	
	# Flatten features for model input
	features_reshaped = features.reshape(features.shape[0] * features.shape[1], -1)
	
	# Generate labels (1 for attacked, 0 for clean) - Example labels for illustration
	labels = np.random.randint(0, 2, features_reshaped.shape[0])
	
	# Train model
	model = train_model(features_reshaped, labels)

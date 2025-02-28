from pathlib import Path
import pandas as pd
import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt
from Attacked_Dataset_Generate import generate_attacked_df
import time


if __name__ == "__main__":
	start_time = time.time()
	continue_attrs = ['temperature_celsius', 'humidity', 'cloud', 'visibility_km', 'uv_index',
	                  'air_quality_Ozone', 'air_quality_Nitrogen_dioxide']  # 连续属性
	discrete_attrs = ['wind_direction', 'air_quality_us-epa-index', 'air_quality_gb-defra-index']  # 离散属性
	drop_column = ['attack_ratio', 'attacked_mode', 'Fault Tolerance']
	result_df = pd.DataFrame()
	
	for _ in range(30):  # 随机10次以去除随机性
		print('第', _, '次')
		generate_attacked_df(1, 31, 0.8, 0.95, 1, discrete_attrs)
		clean_ds_sqr = pd.read_csv('./File/History_Clean_origin_SQR.csv')
		clean_ds_sqr['date'] = pd.to_datetime(clean_ds_sqr['date'], format='%Y-%m-%d', errors='coerce').dt.date
		alpha_df = pd.read_csv(f'./File/alpha_df.csv')
		
		for file in Path('./File/Attacked_Dataset/').glob('*.csv'):
			df = pd.read_csv(file)
			df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce').dt.date
			df = df.fillna(0)
			attacked_feature = os.path.splitext(os.path.basename(file))[0]
			
			for _, row in df.iterrows():
				# 历史干净数据集SQR
				clean_df = clean_ds_sqr[clean_ds_sqr['date'] == row['date']]
				# 计算偏差
				origin_attack_bias = abs(clean_df - row.drop(drop_column)).drop(columns='date', errors='ignore')
				origin_LDP_bias_result = pd.DataFrame()
				for attr in discrete_attrs:  # 计算离散数据的偏度
					matching_columns = [col for col in origin_attack_bias.columns if col.startswith(attr)]
					if matching_columns:
						origin_LDP_bias_result[attr] = origin_attack_bias[matching_columns].sum(axis=1)  # 按行计算这些匹配列的和
				for col in [col for col in origin_attack_bias.columns if  # 给出连续数据的偏度
				            not any(col.startswith(attr) for attr in discrete_attrs)]:
					origin_LDP_bias_result[col] = origin_attack_bias[col]
				
				# 判断偏差是否在阈值范围内
				bias_out_FT_df = pd.DataFrame()
				bias_out_FT_df['detected'] = (
							origin_LDP_bias_result[attacked_feature] >= alpha_df[attacked_feature].values[0]).astype(
					int)
				bias_out_FT_df['date'] = row['date']
				bias_out_FT_df['attack_mode'] = row['attacked_mode']
				bias_out_FT_df['attacked_feature'] = attacked_feature
				bias_out_FT_df['attack_ratio'] = row['attack_ratio']
				
				result_df = pd.concat([result_df, bias_out_FT_df], ignore_index=True)
	
	# 绘图
	fig, axes = plt.subplots(1, 4, figsize=(20, 5))
	
	# 用于存储所有图例句柄和标签
	handles_all = []
	labels_all = []
	
	# 遍历每个 attack_mode
	for i, mode in enumerate(result_df['attack_mode'].unique()):
		ax = axes[i]
		
		# 获取当前 attack_mode 的数据
		mode_df = result_df[result_df['attack_mode'] == mode]
		
		# 遍历每个 attacked_feature
		for file in Path('./File/Attacked_Dataset/').glob('*.csv'):
			attacked_feature = os.path.splitext(os.path.basename(file))[0]
			feature_df = mode_df[mode_df['attacked_feature'] == attacked_feature]
			
			# 按 attack_ratio 分组，计算 detected 不为零的比例
			ratio_counts = feature_df.groupby('attack_ratio')['detected'].agg(lambda x: (x != 0).mean())
			
			# 绘制折线图
			line, = ax.plot(ratio_counts.index, ratio_counts.values, label=attacked_feature)
			# 收集图例句柄和标签
			if i == 0:
				handles_all.append(line)
				labels_all.append(attacked_feature)
		
		# 设置子图标题、坐标轴标签
		ax.set_title(f'Attack Mode: {mode}')
		ax.set_xlabel('Attack Ratio')
		ax.set_ylabel('Success Rate of Detection')
	
	# 在所有子图上方添加一个共同的图例
	fig.legend(handles_all, labels_all, loc='upper center', ncol=len(handles_all), frameon=True,
	           bbox_to_anchor=(0.5, 1))
	# 调整子图布局
	plt.tight_layout(rect=[0, 0, 1, 0.9])
	
	plt.savefig('./File/PDF/Experiment_1_SQR_Bias_In_Attack_Ratio_Change.pdf', format='pdf')
	plt.show()
	
	print(f"代码运行时间: {(int(time.time() - start_time) // 60)} 分钟 {(int(time.time() - start_time) % 60)} 秒")
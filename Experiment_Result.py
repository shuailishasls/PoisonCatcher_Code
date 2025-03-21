from pathlib import Path
import pandas as pd
import numpy as np
import os
import itertools
from tqdm import tqdm
import time
from Attacked_Dataset_Generate import generate_attacked_df, read_files_in_folder, process_date_list, flatten_dict
from Experiment_1_SQR_Bias_In_Attack_Ratio_Change import Experiment_1_SQR_Bias_In_Attack_Ratio_Change
from Experiment_2_SQR_Corr_Analysis import Experiment_2_SQR_Corr_Analysis
from Experiment_3_SQR_Stability_Analysis import Experiment_3_SQR_Stability_Analysis
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import Real_Data_Process as Real_Data_Process
import Statistical as STATI
from Experiment_4_Ratio_Estimates import Experiment_4_Ratio_Estimates


def experiment_1_draw_picture(result_df_1, style_map, file_name):
	fig, axes = plt.subplots(1, 3, figsize=(15, 3))
	plt.rcParams['figure.dpi'] = 1000
	
	# 用于存储所有图例句柄和标签
	handles_all, labels_all = [], []
	
	# 遍历每个 attacked_mode
	for i, mode in enumerate(result_df_1['attacked_mode'].unique()):
		ax = axes[i]
		
		# 获取当前 attacked_mode 的数据
		mode_df = result_df_1[result_df_1['attacked_mode'] == mode]
		
		# 遍历每个 attacked_feature
		for j, file in enumerate(Path('./File/Attacked_Dataset/').glob('*.csv')):
			attacked_feature = os.path.splitext(os.path.basename(file))[0]
			feature_df = mode_df[mode_df['attacked_feature'] == attacked_feature]
			
			# 按 attack_ratio 分组计算F2分数
			ratio_f2 = feature_df.groupby('attack_ratio')[['detection', 'attack_ratio']].apply(STATI.calculate_f2)
			
			# 绘制折线图
			line, = ax.plot(ratio_f2.index, ratio_f2.values, label=attacked_feature,
			                color=style_map[attacked_feature]['color'], marker=style_map[attacked_feature]['marker'],
			                markersize=5, linewidth=1.5, alpha=0.8)
			
			# 收集图例句柄和标签（每个特征只添加一次）
			if i == 0 and attacked_feature not in labels_all:
				handles_all.append(line)
				labels_all.append(attacked_feature)
		
		# 按照 labels_all 的字母顺序对 handles_all 进行排序
		combined = sorted(zip(labels_all, handles_all))
		labels_all, handles_all = [label for label, _ in combined], [handle for _, handle in combined]
		
		# 设置子图属性
		ax.set_title(f'Attack Mode: {mode}')
		ax.set_xlabel('Attack Ratio', fontsize=13)
		ax.set_ylabel('F2 Score', fontsize=13)
		ax.set_ylim(-0.05, 1.05)  # 固定y轴范围
		ax.grid(True, alpha=0.3)
	
	# 添加全局图例
	fig.legend(handles_all, labels_all, loc='upper center', ncol=len(handles_all) / 2, bbox_to_anchor=(0.5, 1))
	
	# 调整布局
	plt.tight_layout(rect=[0, 0, 1, 0.85])
	
	# 保存图片
	plt.savefig(file_name, format='pdf', bbox_inches='tight')
	plt.show()


def experiment_2_draw_picture(result_df_2, style_map, file_name):
	# 获取唯一值列表
	attack_modes = sorted(result_df_2['attacked_mode'].unique())
	features = sorted(result_df_2['attacked_feature'].unique())
	dates = sorted(result_df_2['date'].unique())
	
	# 使用 itertools.product 生成所有组合
	combinations = itertools.product(attack_modes, features, dates)
	new_rows = []
	for mode, feature, date in combinations:
		new_rows.append({'attacked_mode': mode, 'attacked_feature': feature, 'date': date, 'attack_ratio': 0,
		                 'detection': 0})
	new_df = pd.DataFrame(new_rows)
	
	# 合并新 DataFrame 和原 DataFrame
	result_df_2 = pd.concat([result_df_2, new_df], ignore_index=True)
	
	# 创建画布
	fig, axes = plt.subplots(1, 3, figsize=(15, 3))
	plt.rcParams['figure.dpi'] = 1000
	
	# 全局存储图例元素
	legend_handles, legend_labels = [], []
	
	# 绘制每个attack_mode子图
	for ax, mode in zip(axes, attack_modes):
		# 筛选当前模式的数据
		mode_data = result_df_2[result_df_2['attacked_mode'] == mode]
		
		# 绘制每个attacked_feature曲线
		for feature in features:
			feature_data = mode_data[mode_data['attacked_feature'] == feature]
			if not feature_data.empty:
				# 按 attack_ratio 分组计算F2分数
				ratio_f2 = feature_data.groupby('attack_ratio')[['detection', 'attack_ratio']].apply(
					STATI.calculate_f2).reset_index()
				ratio_f2.columns = ['attack_ratio', 'f2_score']
				
				# 绘制曲线
				ax.plot(ratio_f2['attack_ratio'], ratio_f2['f2_score'], label=feature,
				        color=style_map[feature]['color'], marker=style_map[feature]['marker'],
				        markersize=5, linewidth=1.5, alpha=0.8)
				
				# 收集图例元素（避免重复）
				if feature not in legend_labels:
					legend_handles.append(Line2D([0], [0], color=style_map[feature]['color'],
					                             marker=style_map[feature]['marker'], linestyle='-'))
					legend_labels.append(feature)
		
		# 按照 legend_labels 的字母顺序对 legend_handles 进行排序
		combined = sorted(zip(legend_labels, legend_handles))
		legend_labels, legend_handles = [label for label, _ in combined], [handle for _, handle in combined]
		
		# 设置子图属性
		ax.set_title(f'Attack Mode: {mode}')
		ax.set_xlabel('Attack Ratio', fontsize=13)
		ax.set_ylabel('F2 Score', fontsize=13)
		ax.set_ylim(-0.05, 1.05)
		ax.grid(True, alpha=0.3)
	
	# 添加全局图例
	fig.legend(handles=legend_handles, labels=legend_labels, loc='upper center', ncol=len(legend_handles) / 2,
	           bbox_to_anchor=(0.5, 1))
	
	# 保存并显示
	plt.tight_layout(rect=[0, 0, 1, 0.85])
	plt.savefig(file_name, bbox_inches='tight')
	plt.show()


if __name__ == "__main__":
	start_time = time.time()
	continue_attrs = ['temperature_celsius', 'pressure_mb', 'feels_like_celsius', 'gust_kph', 'humidity',
	                  'air_quality_Carbon_Monoxide', 'air_quality_Ozone', 'air_quality_PM10']  # 连续属性
	discrete_attrs = ['air_quality_us-epa-index', 'air_quality_gb-defra-index']  # 离散属性
	times, epsilon, confidence_level, threshold = 32, 1, 0.9, 0.5
	attacked_modes = ['DIPA', 'DPPA', 'ROPA']  # 攻击者的攻击手段

	attack_countries, LDP_sqr, origin_sqr, = [], {}, {}
	result_df_1, result_df_2, result_df_3, result_df_4 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

	country_attacked_ratio = (np.arange(0, 52, 3).astype(float) / 100).tolist()  # 被攻击的国家比例

	Real_Data_Process.process_csv(discrete_attrs, './File/GlobalWeatherRepository.csv', epsilon, times,
	                              continue_attrs + discrete_attrs)  # 数据预处理
	processed_data = pd.read_csv("./File/Preprocessing_Data.csv")  # 读取预处理数据
	processed_data_non_LDP = pd.read_csv("./File/Preprocessing_Data(non_LDP).csv")  # 读取预处理数据
	dfs_LDP_dict = read_files_in_folder('./File/Divide_data_by_time')  # 按日读取文件夹的内容存成一个dataframe
	dfs_dict = read_files_in_folder('./File/Divide_data_by_time(non_LDP)')

	# 确定攻击dates
	history_ds, current_ds = process_date_list(pd.to_datetime(processed_data['date'].unique()).date, threshold)

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

	# 对原始数据集中的每个属性计算统计结果
	for key, df in dfs_dict.items():
		origin_sqr[key] = STATI.calculate_features_statistics(df.iloc[:, 2:], discrete_attrs, epsilon)
	origin_sqr = flatten_dict(origin_sqr, discrete_attrs)  # 将输入的3维字典2维化处理

	# 计算原始数据与LDP处理后的数据的偏差
	origin_LDP_bias_result = STATI.compute_bias(discrete_attrs, (LDP_sqr.iloc[:, 1:] - origin_sqr.iloc[:, 1:]).abs())

	# 计算基线数据集中每个属性的容错阈值
	alpha_df = STATI.calculate_ft(origin_sqr.iloc[:, 1:], len(dfs_dict[next(iter(dfs_dict))]), epsilon,
	                              confidence_level, discrete_attrs).T
	# 判断偏差是否在阈值范围内
	bias_out_FT_df = origin_LDP_bias_result.copy()
	for col in origin_LDP_bias_result.columns:
		bias_out_FT_df[col] = (origin_LDP_bias_result[col] > alpha_df[col].values[0]).astype(int)
		if bias_out_FT_df[col].sum() != 0:
			print(col)

	# ---------------------------开始实验---------------------------
	for _ in tqdm(range(30)):  # 随机多次以去除随机性
		_ = [f.unlink() for f in Path('./File/Attacked_Dataset').iterdir() if f.is_file()]  # 清空文件夹

		# 攻击数据生成
		generate_attacked_df(epsilon, discrete_attrs, attack_countries, continue_attrs + discrete_attrs, attacked_modes,
		                     country_attacked_ratio, current_ds, dfs_LDP_dict, dfs_dict, alpha_df, true_attack_ratio,
		                     processed_data, processed_data_non_LDP)

		result_df_temp1, result_df_bias_1 = Experiment_1_SQR_Bias_In_Attack_Ratio_Change(discrete_attrs, origin_sqr.copy(), alpha_df)
		result_df_1 = pd.concat([result_df_1, result_df_temp1], ignore_index=True)

		result_df_temp2, result_df_bias_2 = Experiment_2_SQR_Corr_Analysis(continue_attrs, discrete_attrs, origin_sqr.copy(), confidence_level)
		result_df_2 = pd.concat([result_df_2, result_df_temp2], ignore_index=True)

		# 提取攻击程度、攻击模式、攻击特征相同，时间不同作为一组，进行方差、波动、自相关性判断
		result_df_temp3_1 = Experiment_3_SQR_Stability_Analysis(result_df_bias_1)
		result_df_temp3_2 = Experiment_3_SQR_Stability_Analysis(result_df_bias_2)
		result_df_3 = pd.concat([result_df_3, result_df_temp3_1, result_df_temp3_2], ignore_index=True)
		
		result_df_4 = pd.concat([result_df_3, result_df_bias_2, result_df_bias_1], ignore_index=True)

	result_df_1.to_csv(f'./File/Experiment_1_SQR_Bias_In_Attack_Ratio_Change.csv', index=False)
	result_df_2.to_csv(f'./File/Experiment_2_SQR_Corr_Analysis.csv', index=False)
	result_df_3.to_csv(f'./File/Experiment_3_SQR_Stability_Analysis.csv', index=False)
	result_df_4.to_csv(f'./File/Experiment_4_Ratio_Estimates.csv', index=False)

	# result_df_1 = pd.read_csv(f'./File/Experiment_1_SQR_Bias_In_Attack_Ratio_Change.csv')
	# result_df_2 = pd.read_csv(f'./File/Experiment_2_SQR_Corr_Analysis.csv')
	# result_df_3 = pd.read_csv(f'./File/Experiment_3_SQR_Stability_Analysis.csv')

	colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:orange', 'tab:purple', 'tab:brown']  # 定义 10 种不同的颜色
	markers = ['o', 's', '^', 'D', 'v', '*', 'p', 'h', '8', 'x']  # 定义 10 种不同的标记
	style_map = {feat: {'color': colors[i % len(colors)], 'marker': markers[i % len(markers)], 'linestyle': '-'}
	             for i, feat in enumerate(sorted(result_df_1['attacked_feature'].unique()))}

	experiment_1_draw_picture(result_df_1, style_map, './File/PDF/Experiment_1_F2_Score_vs_Attack_Ratio.pdf')  # 实验1绘图
	experiment_2_draw_picture(result_df_2, style_map, './File/PDF/Experiment_2_SQR_Corr_Analysis.pdf')  # 实验2绘图
	experiment_1_draw_picture(result_df_3, style_map, './File/PDF/Experiment_3_SQR_Stability_Analysis.pdf')  # 实验3绘图
	Experiment_4_Ratio_Estimates(processed_data_non_LDP, result_df_4)
	
	elapsed_seconds = time.time() - start_time
	print(f"代码运行时间: {int(elapsed_seconds // 3600):02d}小时{int((elapsed_seconds % 3600) // 60):02d}分钟")

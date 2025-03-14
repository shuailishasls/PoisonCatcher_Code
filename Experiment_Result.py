from pathlib import Path
import pandas as pd
import numpy as np
import os
import itertools
from tqdm import tqdm
import time
from Statistic_In_Experiment import calculate_f2
from Attacked_Dataset_Generate import generate_attacked_df
from Experiment_1_SQR_Bias_In_Attack_Ratio_Change import Experiment_1_SQR_Bias_In_Attack_Ratio_Change
from Experiment_2_SQR_Corr_Analysis import Experiment_2_SQR_Corr_Analysis
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def experiment_1_draw_picture(result_df_1, style_map):
	fig, axes = plt.subplots(1, 3, figsize=(15, 5))
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
			ratio_f2 = feature_df.groupby('attack_ratio')[['detection', 'attack_ratio']].apply(calculate_f2)
			
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
		ax.set_xlabel('Attack Ratio')
		ax.set_ylabel('F2 Score')
		ax.set_ylim(-0.05, 1.05)  # 固定y轴范围
		ax.grid(True, alpha=0.3)
	
	# 添加全局图例
	fig.legend(handles_all, labels_all, loc='upper center', ncol=len(handles_all) / 2, bbox_to_anchor=(0.5, 1))
	
	# 调整布局
	plt.tight_layout(rect=[0, 0, 1, 0.9])
	
	# 保存图片
	plt.savefig('./File/PDF/Experiment_1_F2_Score_vs_Attack_Ratio.pdf', format='pdf', bbox_inches='tight')
	plt.show()


def experiment_2_draw_picture(result_df_2, style_map):
	# 获取唯一值列表
	attack_modes = sorted(result_df_2['attacked_mode'].unique())
	features = sorted(result_df_2['attacked_feature'].unique())
	dates = sorted(result_df_2['date'].unique())
	
	# 使用 itertools.product 生成所有组合
	combinations = itertools.product(attack_modes, features, dates)
	new_rows = []
	for mode, feature, date in combinations:
		new_rows.append({'attacked_mode': mode, 'attacked_feature': feature, 'date': date, 'attack_ratio': 0,
		                 'successful_detection': 0})
	new_df = pd.DataFrame(new_rows)
	
	# 合并新 DataFrame 和原 DataFrame
	result_df_2 = pd.concat([result_df_2, new_df], ignore_index=True)
	
	# 创建画布
	fig, axes = plt.subplots(1, 3, figsize=(15, 5))
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
				ratio_f2 = feature_data.groupby('attack_ratio')[['successful_detection', 'attack_ratio']].apply(
					calculate_f2).reset_index()
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
		ax.set_xlabel('Attack Ratio')
		ax.set_ylabel('F2 Score')
		ax.set_ylim(-0.05, 1.05)
		ax.grid(True, alpha=0.3)
	
	# 添加全局图例
	fig.legend(handles=legend_handles, labels=legend_labels, loc='upper center', ncol=len(legend_handles) / 2,
	           bbox_to_anchor=(0.5, 1))
	
	# 保存并显示
	plt.tight_layout(rect=[0, 0, 1, 0.9])
	plt.savefig('./File/PDF/Experiment_2_SQR_Corr_Analysis.pdf', bbox_inches='tight')
	plt.show()


if __name__ == "__main__":
	start_time = time.time()
	continue_attrs = ['temperature_celsius', 'pressure_mb', 'feels_like_celsius', 'gust_kph', 'humidity',
	                  'air_quality_Carbon_Monoxide', 'air_quality_Ozone', 'air_quality_PM10']  # 连续属性
	discrete_attrs = ['air_quality_us-epa-index', 'air_quality_gb-defra-index']  # 离散属性
	drop_column = ['attack_ratio', 'attacked_mode', 'Fault Tolerance', 'date']
	# result_df_1 = pd.DataFrame()
	# result_df_2 = pd.DataFrame()

	for _ in tqdm(range(300)):  # 随机多次以去除随机性  实验2跑1次需要3分半
		print('第', _, '次')

		_ = [f.unlink() for f in Path('./File/Attacked_Dataset').iterdir() if f.is_file()]

		generate_attacked_df(1, 32, 0.8, 0.9, 1, discrete_attrs, continue_attrs)
		clean_ds_sqr = pd.read_csv('./File/History_Clean_origin_SQR.csv')
		clean_ds_sqr['date'] = pd.to_datetime(clean_ds_sqr['date'], format='%Y-%m-%d', errors='coerce').dt.date
		# alpha_df = pd.read_csv(f'./File/alpha_df.csv')
		
		# 	result_df_temp1 = Experiment_1_SQR_Bias_In_Attack_Ratio_Change(discrete_attrs, drop_column, clean_ds_sqr, alpha_df)
		# 	result_df_1 = pd.concat([result_df_1, result_df_temp1], ignore_index=True)
		
		# result_df_temp2 = Experiment_2_SQR_Corr_Analysis(continue_attrs, discrete_attrs, drop_column, clean_ds_sqr, 0.95)
		# result_df_2 = pd.concat([result_df_2, result_df_temp2], ignore_index=True)
	
	# result_df_1.to_csv(f'./File/Experiment_1_SQR_Bias_In_Attack_Ratio_Change.csv', index=False)
	# result_df_2.to_csv(f'./File/Experiment_2_SQR_Corr_Analysis.csv', index=False)
	
	# result_df_1 = pd.read_csv(f'./File/Experiment_1_SQR_Bias_In_Attack_Ratio_Change.csv')
	result_df_2 = pd.read_csv(f'./File/Experiment_2_SQR_Corr_Analysis.csv')
	
	colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:orange', 'tab:purple', 'tab:brown']  # 定义 10 种不同的颜色
	markers = ['o', 's', '^', 'D', 'v', '*', 'p', 'h', '8', 'x']  # 定义 10 种不同的标记
	style_map = {feat: {'color': colors[i % len(colors)], 'marker': markers[i % len(markers)], 'linestyle': '-'}
	             for i, feat in enumerate(sorted(result_df_2['attacked_feature'].unique()))}
	
	# experiment_1_draw_picture(result_df_1, style_map)  # 实验1绘图
	experiment_2_draw_picture(result_df_2, style_map)  # 实验2绘图
	
	elapsed_seconds = time.time() - start_time
	hours = int(elapsed_seconds // 3600)
	minutes = int((elapsed_seconds % 3600) // 60)
	seconds = int(elapsed_seconds % 60)
	print(f"代码运行时间: {hours:02d}小时{minutes:02d}分钟{seconds:02d}秒")

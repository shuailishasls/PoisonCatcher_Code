import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerLine2D


def extract_data(folder_path, new_name):
	data_frames = []
	# 遍历指定文件夹中的所有文件
	for file_name in os.listdir(folder_path):
		# 检查文件是否为 CSV 文件
		if file_name.endswith('.csv'):
			file_path = os.path.join(folder_path, file_name)
			try:
				df = pd.read_csv(file_path)
				# 提取 1、2、4 列
				new_df = df[['attacked_feature', 'attacked_mode', 'F2 score']]
				# 可以根据需要从文件名中提取时间实例等信息，这里简单使用文件名作为标识
				new_df[new_name] = float(os.path.splitext(file_name)[0])
				data_frames.append(new_df)
			except Exception as e:
				print(f"读取文件 {file_path} 时出错: {e}")
	
	# 合并所有 DataFrame
	if data_frames:
		result_df = pd.concat(data_frames, ignore_index=True)
		return result_df
	else:
		print("未找到有效的 CSV 文件。")
		return None


def experiment_6_draw_picture(result_dfs, style_map, x_low, file_name, x_name):
	"""
	根据攻击模式绘制 F2 分数的子图
	:param result_dfs: 包含多个 DataFrame 的列表，每个 DataFrame 包含特征名、时间、attacked_mode、F2 score
	"""
	if x_name == 'Attack Ratio':
		group_by_name = 'attack_ratio'
	elif x_name == 'Time Instance':
		group_by_name = 'time_instance'
	else:
		group_by_name = 'epsilon'
	# 按攻击模式分组
	all_grouped = []
	for result_df in result_dfs:
		grouped = result_df.groupby('attacked_mode')
		all_grouped.append(grouped)
	
	# 创建包含三个子图的图形
	fig, axes = plt.subplots(1, 3, figsize=(15, 2.5))
	plt.rcParams['figure.dpi'] = 1000
	
	legend_handles, legend_labels = [], []
	
	for i in range(3):
		ax = axes[i]
		for j, grouped in enumerate(all_grouped):
			attack_mode, group = list(grouped)[i]
			for attacked_feature, time_f2 in group.groupby('attacked_feature'):
				time_f2 = time_f2.groupby(group_by_name)['F2 score'].mean()
				line, = ax.plot(time_f2.index, time_f2.values, label=attacked_feature,
				                color=style_map[attacked_feature]['color'],
				                marker=style_map[attacked_feature]['marker'],
				                markersize=5, linewidth=1.5, alpha=0.8, linestyle=['-', '--'][j])
				# 收集图例元素（避免重复）
				if attacked_feature not in legend_labels:
					legend_handles.append(Line2D([0], [0], color=style_map[attacked_feature]['color'],
					                             marker=style_map[attacked_feature]['marker']))
					legend_labels.append(attacked_feature)
		
		# 按照 legend_labels 的字母顺序对 legend_handles 进行排序
		combined = sorted(zip(legend_labels, legend_handles))
		legend_labels, legend_handles = [label for label, _ in combined], [handle for _, handle in combined]
		
		# 设置子图属性
		ax.set_title(f'Attack Mode: {attack_mode}')
		ax.set_xlabel(x_name, fontsize=13)
		ax.set_ylabel('F2 Score', fontsize=13)
		ax.set_ylim(x_low, 1.05)
		ax.grid(True, alpha=0.3)
	
	# 自定义图例处理函数，只显示标记
	def update(handle, orig):
		handle.set_linestyle("None")
		handle.set_marker(orig.get_marker())
		handle.set_markersize(orig.get_markersize())
		handle.set_color(orig.get_color())
	
	# 添加第一个全局图例（显示标记对应的特征）
	first_legend = fig.legend(handles=legend_handles, labels=legend_labels, loc='upper center',
	                          ncol=len(legend_handles) // 2,
	                          bbox_to_anchor=(0.45, 1), handler_map={Line2D: HandlerLine2D(update_func=update)})
	
	# 创建第二个图例，说明实线和虚线分别是什么
	line_handles = [Line2D([0], [0], color='black', linestyle='-'),
	                Line2D([0], [0], color='black', linestyle='--')]
	line_labels = ['FE-enhanced RF', 'Baseline RF']
	fig.legend(handles=line_handles, labels=line_labels, loc='upper center', ncol=1, bbox_to_anchor=(0.89, 1))
	
	# 将第一个图例添加回图形
	fig.add_artist(first_legend)
	
	plt.tight_layout(rect=[0, 0, 1, 0.8])
	plt.savefig(file_name, bbox_inches='tight')
	plt.show()


result_df_5 = pd.read_csv(f'./File/Experiment_5_Identify_Attacked_Data.csv')
result_df_5_1 = pd.read_csv(f'./File/Experiment_5_1_Identify_Attacked_Data.csv')

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:orange', 'tab:purple', 'tab:brown']  # 定义 10 种不同的颜色
markers = ['o', 's', '^', 'D', 'v', '*', 'p', 'h', '8', 'x']  # 定义 10 种不同的标记
style_map = {feat: {'color': colors[i % len(colors)], 'marker': markers[i % len(markers)], 'linestyle': '-'}
             for i, feat in enumerate(sorted(result_df_5['attacked_feature'].unique()))}

# 绘制图形
experiment_6_draw_picture([result_df_5, result_df_5_1], style_map, 0,
                          './File/PDF/Experiment_5_Identify_Attacked_Data.pdf', 'Attack Ratio')

# 提取数据  time_instance_vs_f2_score
result_df_6 = extract_data('./File/Experiment_6/time_instance_vs_f2_score/Experiment_6', 'time_instance')
result_df_6_1 = extract_data('./File/Experiment_6/time_instance_vs_f2_score/Experiment_6_1', 'time_instance')

# 绘制图形
experiment_6_draw_picture([result_df_6, result_df_6_1], style_map, 0.8,
                          './File/PDF/Experiment_6_time_instance_vs_f2_score.pdf', 'Time Instance')

# 提取数据  epsilon_vs_f2_score
result_df_6 = extract_data('./File/Experiment_6/epsilon_vs_f2_score/Experiment_6', 'epsilon')
result_df_6_1 = extract_data('./File/Experiment_6/epsilon_vs_f2_score/Experiment_6_1', 'epsilon')

# 绘制图形
experiment_6_draw_picture([result_df_6, result_df_6_1], style_map, 0.9,
                          './File/PDF/Experiment_6_epsilon_vs_f2_score.pdf', 'Epsilon')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('../File/Experiment_2_Spearman_Corr_sqr.csv')
attacked_modes, result, date = ['RPVA', 'RIA', 'RPA', 'MGA'], [], '2024-12-26'
clean_df = df[(df['attack_ratio'] == 0) & (df['attacked_mode'] == 'Clean')].iloc[:, :-3]  # 干净SQR
corr_matrix_original = clean_df.corr()  # 计算相关性矩阵

for j in attacked_modes:
	attacked_ratio_list = df[(df['date'] == date) & (df['attacked_mode'] == j)]['attack_ratio'].unique()
	for i in attacked_ratio_list:
		# 第二步：找到符合条件的一行数据并去掉最后3列
		extra_row = df[(df['attack_ratio'] == i) & (df['attacked_mode'] == j) & (df['date'] == date)].iloc[:, :-3]
		new_df = pd.concat([clean_df, extra_row], ignore_index=True)
		# 计算相关性矩阵
		corr_matrix_new = new_df.corr()
		# 计算两个矩阵的差异（对应元素差值的绝对值）
		diff_matrix = np.abs(corr_matrix_original - corr_matrix_new)
		# 计算差异矩阵的均值和最大值（作为变化程度的衡量）
		mean_diff, max_diff = np.mean(diff_matrix), np.max(diff_matrix)
		result.append([j, i, mean_diff, max_diff])

df = pd.DataFrame(result, columns=['attacked_mode', 'attack_ratio', 'mean_diff', 'max_diff'])
# 将 attack_ratio 保留两位小数
df['attack_ratio'] = df['attack_ratio'].round(2)

attack_ratios = df[df['attacked_mode'] == 'RPVA']['attack_ratio'].unique()

# ---------------------- 绘图 ----------------------
fig, ax1 = plt.subplots(figsize=(10, 6))
# 获取 viridis 颜色映射
colormap = plt.colormaps.get_cmap('viridis')
# 重新采样颜色映射，指定颜色数量为 4
colormap = colormap.resampled(4)
# 生成颜色列表
colors = [colormap(i) for i in np.linspace(0, 1, 4)]

# 计算每个柱状图组的中心位置，以保证对齐
x_centers = attack_ratios
for i, mode in enumerate(attacked_modes):
	mode_data = df[df['attacked_mode'] == mode]
	# 修改柱状图的x位置，使用计算出的中心位置
	ax1.bar(x_centers + (i - len(attacked_modes) / 2 + 0.5) * 0.008, mode_data['mean_diff'],
	        width=0.008, color=colors[i], label=f'Mean Diff - {mode}', alpha=0.7)

# 创建第二个y轴，绘制折线图：以attack_ratio为x轴，max_diff为y轴
ax2 = ax1.twinx()

for i, mode in enumerate(attacked_modes):
	mode_data = df[df['attacked_mode'] == mode]
	ax2.plot(mode_data['attack_ratio'], mode_data['max_diff'],
	         color=colors[i], label=f'Max Diff - {mode}', linestyle='-', linewidth=2)

# 设置标签
ax1.set_xlabel('Actual Attack Ratio')
ax1.set_ylabel('Mean Diff')
ax2.set_ylabel('Max Diff')
ax2.set_ylim(0, 2)

# 设置x轴刻度
ax1.set_xticks(attack_ratios)  # 使用attack_ratio值作为x轴刻度
ax1.set_xticklabels(attack_ratios)

# 添加图例，放置在图片外面
lines, labels = ax2.get_legend_handles_labels()
bars, bar_labels = ax1.get_legend_handles_labels()
ax1.legend(bars + lines, bar_labels + labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4)
plt.tight_layout()

plt.savefig('../File/PDF/Experiment_2_Attack_Impact_Attribute_Correlation.pdf', format='pdf')
# 显示图形
plt.show()

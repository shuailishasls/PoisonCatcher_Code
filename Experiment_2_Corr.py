import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


# 对 target_feature 和 other_feature 进行处理
def process_feature(feature):
	parts = feature.split('_')
	return ' '.join([part.capitalize() for part in parts])


# 加载数据，这里路径修改为你上传文件的路径
df = pd.read_csv('../File/Experiment_2_Spearman_Corr_sqr.csv')

df['target_feature'] = df['target_feature'].apply(process_feature)
df['other_feature'] = df['other_feature'].apply(process_feature)

# 设置全局字体大小
plt.rcParams['font.size'] = 12  # 你可以根据需要调整这个值

# 构建横坐标标签
df['combined_feature'] = (df['real_attacked_ratio'].astype(str) + '\n' + df['expected_attack_ratio'].astype(str))

# 获取 unique 的 target_feature 和 other_feature 以及 attacked_mode
unique_target_features = df['target_feature'].unique()
unique_other_features = df['other_feature'].unique()
unique_attacked_modes = df['attacked_mode'].unique()

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'

# 创建两个大的子图组，每个组包含 4 个子图（假设 attacked_mode 有 4 种）
fig, axes = plt.subplots(2, len(unique_attacked_modes), figsize=(30, 10), sharex=True, sharey=False)

# 定义单一颜色和透明度范围
base_color = (0, 0, 1)  # 这里使用蓝色为例，你可以根据需要修改
transparency_range = [(0, 0, 0, 0), base_color + (1,)]  # 从完全透明到不透明

# 创建自定义颜色映射
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', transparency_range)

# 获取全局最小值和最大值，确保所有子图使用相同的颜色范围
min_corr = df['correlation'].min()
max_corr = df['correlation'].max()

for i, target_feature in enumerate(unique_target_features):
	for j, attacked_mode in enumerate(unique_attacked_modes):
		subset = df[(df['target_feature'] == target_feature) & (df['attacked_mode'] == attacked_mode)]
		pivot_table = subset.pivot_table(index='other_feature', columns='combined_feature', values='correlation')
		
		# 计算每行从第二列开始与第一列的差值
		diff_pivot_table = pivot_table.sub(pivot_table.iloc[:, 0], axis=0).drop(columns=pivot_table.columns[0])
		diff_pivot_table = diff_pivot_table.round(4)
		diff_pivot_table = abs(diff_pivot_table)
		
		# 创建和 pivot_table 形状相同的 DataFrame，第一列填充为 NaN
		annot_data = pd.DataFrame(index=pivot_table.index, columns=pivot_table.columns)
		annot_data.iloc[:, 0] = ''
		annot_data.iloc[:, 1:] = diff_pivot_table
		
		ax = axes[i, j]
		# 不显示每个子图的颜色条
		sns.heatmap(pivot_table, annot=False, cmap=custom_cmap, ax=ax, vmin=min_corr, vmax=max_corr, cbar=False,
		            annot_kws={'color': 'black'}, fmt='')
		ax.set_title(f'{attacked_mode} in {target_feature}')
		if i == len(unique_target_features) - 1:
			ax.set_xlabel('')
			# 设置 x 轴标签旋转角度为 0 度
			plt.setp(ax.get_xticklabels(), rotation=0)
			# 确保 x 轴刻度线和标签显示
			ax.tick_params(axis='x', bottom=True, labelbottom=True)
		if j != 0:
			ax.tick_params(axis='y', labelleft=False)
		ax.set_ylabel('')
		ax.set_xlabel('')

# 在整个图形的 x 轴最左边标注 real attacked ratio，expected attack ratio，attacked mode
fig.text(0.05, 0.088, 'Real Attacked Ratio\nExpected Attack Ratio', ha='left', va='top', fontsize=10,
         bbox=dict(facecolor='lightgray', boxstyle='round,pad=0.3'))

# 添加一个共同的颜色条，设置其跨越两行子图
# 调整颜色条的位置，使其不与子图重合
sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=min_corr, vmax=max_corr))
sm.set_array([])

# 调整颜色条位置
cbar_ax = fig.add_axes([0.92, 0.1, 0.01, 0.8])  # 右侧，跨越两个子图行
cbar = fig.colorbar(sm, cax=cbar_ax)

# 调整颜色条刻度文字的大小
cbar.ax.tick_params(labelsize=14)  # 将刻度文字大小设置为 12

# 使用 adjust_subplots 来微调子图的布局，避免重叠
plt.subplots_adjust(right=0.90, top=0.9, bottom=0.1, left=0.1)

# 保存PDF时去除白边，bbox_inches='tight' 让Matplotlib自动裁剪白边
plt.savefig('../File/PDF/Experiment_2_Spearman_Corr.pdf', format='pdf', bbox_inches='tight', pad_inches=0.1)

# 显示图形
plt.show()

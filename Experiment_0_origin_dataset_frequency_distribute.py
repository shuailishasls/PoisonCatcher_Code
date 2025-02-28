# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Set color palette
sns.set_palette("Set2")

# Set the option to display the maximum number of column
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# 读取数据
df = pd.read_csv("./File/Preprocessing_Data(non_LDP).csv")

# Specify the columns to visualize
selected_columns = ['temperature_celsius', 'wind_kph', 'pressure_mb', 'precip_mm', 'humidity', 'cloud',
                    'feels_like_celsius', 'visibility_km', 'uv_index', 'gust_kph', 'air_quality_Carbon_Monoxide',
                    'air_quality_Ozone', 'air_quality_Nitrogen_dioxide', 'air_quality_Sulphur_dioxide',
                    'air_quality_PM2.5', 'air_quality_PM10', 'condition_text', 'wind_direction',
                    'air_quality_us-epa-index', 'air_quality_gb-defra-index']

# Create a figure for the subplots
n_cols = 5
n_rows = (len(selected_columns) + n_cols - 1) // n_cols  # Calculate number of rows needed
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easy indexing

# Get the SET2 color palette
set2_palette = sns.color_palette("Set2", n_colors=len(selected_columns))

# Loop through the selected columns and create plots
for i, col in enumerate(selected_columns):
	if pd.api.types.is_numeric_dtype(df[col]):
		sns.histplot(df[col], ax=axes[i], bins=30, kde=True, color=set2_palette[i])
		axes[i].set_xlabel(col)
		axes[i].set_ylabel('Frequency')
	else:
		# 对于非数值型数据，使用计数图
		sns.countplot(x=col, data=df, ax=axes[i], color=set2_palette[i])
		axes[i].set_xlabel(col)
		axes[i].set_ylabel('Count')
		
		# 获取 x 轴标签
		x_ticks = axes[i].get_xticks()
		x_ticklabels = axes[i].get_xticklabels()
		
		# 如果 x 轴标签数量超过 10 个，则跳着显示
		if len(x_ticklabels) > 10:
			step = 2  # 可以根据实际情况调整步长
			new_x_ticks = x_ticks[::step]
			new_x_ticklabels = x_ticklabels[::step]
			axes[i].set_xticks(new_x_ticks)
			axes[i].set_xticklabels(new_x_ticklabels)
		
		# 旋转 x 轴标签以避免重叠
		axes[i].tick_params(axis='x', rotation=90)

# Hide any unused subplots
for j in range(len(selected_columns), len(axes)):
	fig.delaxes(axes[j])

# Adjust layout
plt.tight_layout()
plt.savefig('./File/PDF/Experiment_0_Processed_Data_Distribution.pdf', format='pdf')
plt.show()

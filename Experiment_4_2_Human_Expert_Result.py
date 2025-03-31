import os
import numpy as np
import pandas as pd

# 定义根文件夹路径
root_folder = './File/Human_Expert_Review_Scale_result'

# 处理名称为 00 的子文件夹，该文件夹中包含真实攻击比例
sub_folder_00 = os.path.join(root_folder, '00')
# 处理非 00 的子文件夹，改文件夹为专家人工评估结果
other_sub_folders = [f for f in os.listdir(root_folder) if f != '00']
discrete_attrs = ['air_quality_us-epa-index', 'air_quality_gb-defra-index']  # 离散属性

discrete_attack_ratios = pd.DataFrame()
continuity_attack_ratios = pd.DataFrame()
for file_name in os.listdir(sub_folder_00):
	combined_df = pd.DataFrame()
	
	df = pd.read_csv(os.path.join(sub_folder_00, file_name))
	combined_df = pd.concat([combined_df, df[['attacked_mode', 'attack_ratio']]], axis=1)
	
	for sub_folder in other_sub_folders:
		df = pd.read_csv(os.path.join(root_folder, sub_folder, file_name))
		combined_df = pd.concat([combined_df, df['attack_ratio']], axis=1)
	
	average = combined_df.iloc[:, 2:].mean(axis=1)
	result = combined_df.iloc[:, 2:].mean(axis=1) / (combined_df.iloc[:, 1] / 0.05)
	result = pd.concat((combined_df.iloc[:, 0], result), axis=1)
	final_result = result[~result.iloc[:, 1].isna()].groupby(result.iloc[:, 0])[result.columns[1]].mean()
	if file_name.replace('.csv', '') in discrete_attrs:
		discrete_attack_ratios = pd.concat([discrete_attack_ratios, final_result], axis=1)
	else:
		continuity_attack_ratios = pd.concat([continuity_attack_ratios, final_result], axis=1)
		
continuity_attack_ratios = continuity_attack_ratios.replace([np.inf, -np.inf], 0)
discrete_attack_ratios = discrete_attack_ratios.replace([np.inf, -np.inf], 0)
print('连续属性人类专家估计值：\n', continuity_attack_ratios.mean(axis=1))
print('离散属性人类专家估计值：\n', discrete_attack_ratios.mean(axis=1))

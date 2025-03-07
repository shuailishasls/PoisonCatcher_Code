import os
import seaborn as sns
from pathlib import Path
import pandas as pd
import numpy as np
import Statistical as STATI
from pathlib import Path


def Experiment_1_SQR_Bias_In_Attack_Ratio_Change(discrete_attrs, drop_column, clean_ds_sqr, alpha_df):
	result_df = pd.DataFrame()

	for file in Path('./File/Attacked_Dataset/').glob('*.csv'):
		df = pd.read_csv(file)
		df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce').dt.date
		df = df.fillna(0)
		attacked_feature = os.path.splitext(os.path.basename(file))[0]

		for _, row in df.iterrows():
			# 历史干净数据集SQR
			clean_df = clean_ds_sqr[clean_ds_sqr['date'] == row['date']]
			origin_attack_bias = abs(clean_df - row.drop(drop_column)).drop(columns='date', errors='ignore')
			origin_LDP_bias_result = STATI.compute_bias(discrete_attrs, origin_attack_bias)

			# 判断偏差是否在阈值范围内
			bias_out_FT_df = pd.DataFrame()
			bias_out_FT_df['detection'] = (
					origin_LDP_bias_result[attacked_feature] >= alpha_df[attacked_feature].values[0]).astype(
				int)
			bias_out_FT_df['date'] = row['date']
			bias_out_FT_df['attacked_mode'] = row['attacked_mode']
			bias_out_FT_df['attacked_feature'] = attacked_feature
			bias_out_FT_df['attack_ratio'] = row['attack_ratio']

			result_df = pd.concat([result_df, bias_out_FT_df], ignore_index=True)
	return result_df

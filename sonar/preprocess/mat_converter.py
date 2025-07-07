"""
File Name: mat_converter.py
Author: Yixiao Shen
Date: 2025-05-26
Purpose: converter for .mat format from ZYC
"""
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat

from sonar.core.region_selector import RegionSelector
from sonar.preprocess.normalization import z_score_normalization

def mat_converter(mat_path, sub_label_l, time_shifting=0, crop_dict=None, new_names=None):
	# Load .mat file
	data = loadmat(mat_path)
	homer_data = data['Homer_Data'][0]
	n_sub = len(homer_data)
	assert n_sub == len(sub_label_l), "Mismatch between number of subjects and sub_dict"

	# Get parent directory of the .mat file
	parent_dir = os.path.dirname(mat_path)
	hbo_raw_dir = os.path.join(parent_dir, 'hbo_raw')
	hbo_normalized_dir = os.path.join(parent_dir, 'hbo')

	os.makedirs(hbo_raw_dir, exist_ok=True)
	os.makedirs(hbo_normalized_dir, exist_ok=True)

	for sub_idx, sub_label in enumerate(sub_label_l):
		if sub_label is None:
			continue
		region_selector: RegionSelector = crop_dict[sub_label]
		subject_data = homer_data[sub_idx][0]
		time = subject_data[0]  # shape: (T,)
		hbo = subject_data[1]  # shape: (T, 48)

		# Ensure time is column vector
		time = time.reshape(-1, 1)
		hbo_with_time = np.hstack([hbo, time])  # shape: (T, 49)

		# Column names: ch1, ch2, ..., ch48, time
		col_names = [f'ch{i+1}' for i in range(hbo.shape[1])] + ['time']

		# Save to CSV
		df = pd.DataFrame(hbo_with_time, columns=col_names)
		output_path = os.path.join(hbo_raw_dir, f'{sub_label}.csv')

		# Apply unit conversion
		for col in df.columns:
			if col != "time":  # Skip the "time" column
				# 原始单位为mmol/L
				df[col] *= 1e3  # Multiply by 1e3, 单位变成umol/L

		# Crop
		df_crop = df[(df["time"] >= region_selector.start_sec) & (df["time"] <= region_selector.end_sec)].copy()

		# Adjust time: subtract first timestamp and add shifting
		df_crop["time"] = df_crop["time"] - df_crop["time"].iloc[0] + time_shifting

		if new_names is not None:
			df_crop.columns = new_names

		# Save to CSV
		df_crop.to_csv(output_path, index=False)
		print(f"Saved: {output_path}")


	z_score_normalization(src_dir=hbo_raw_dir, tar_dir=hbo_normalized_dir)

if __name__ == '__main__':
	mat_path = 'res/trainingcamp-homer3/homerdata.mat'
	sub_dict =[
		None,
		None,
		None,
		None,
		None,
		'HC3',
		'HC5',
		'HC1',
		'HC9',
		'HC7',
	]
	crop_dict = {}
	crop_dict['HC1']=RegionSelector(start_sec=2055.455, end_sec=4817.909)
	crop_dict['HC3']=RegionSelector(start_sec=2052.727, end_sec=4815.000)
	crop_dict['HC5']=RegionSelector(start_sec=2061.727, end_sec=4824.091)
	crop_dict['HC7']=RegionSelector(start_sec=95.182, end_sec=2857.727)
	crop_dict['HC9']=RegionSelector(start_sec=117.273, end_sec=2879.727)

	
	mat_converter(mat_path, sub_label_l=sub_dict, crop_dict=crop_dict)
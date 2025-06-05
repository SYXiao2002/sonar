"""
File Name: mat_converter.py
Author: Yixiao Shen
Date: 2025-05-26
Purpose: converter for .mat format from ZYC
"""
import os
from re import sub
import numpy as np
import pandas as pd
from scipy.io import loadmat

def mat_converter(mat_path, sub_label_l, time_shifting=0):
	# Load .mat file
	data = loadmat(mat_path)
	homer_data = data['Homer_Data'][0]
	n_sub = len(homer_data)
	assert n_sub == len(sub_label_l), "Mismatch between number of subjects and sub_dict"

	# Get parent directory of the .mat file
	parent_dir = os.path.dirname(mat_path)
	output_dir = os.path.join(parent_dir, 'hbo')
	os.makedirs(output_dir, exist_ok=True)

	# Create hbo/ folder if not exists
	os.makedirs(output_dir, exist_ok=True)

	for sub_idx, sub_label in enumerate(sub_label_l):
		if sub_label is None:
			continue
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
		output_path = os.path.join(output_dir, f'{sub_label}.csv')

		df['time'] = df['time'] + time_shifting

		# Save back to CSV (optional)
		df.to_csv(output_path, index=False)

		print(f"Saved: {output_path}")


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
	mat_converter(mat_path, sub_label_l=sub_dict)
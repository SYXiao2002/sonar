import csv
import os

import pandas as pd

from sonar.preprocess.normalization import z_score_normalization


def nirspark2csv(ds_dir, sr, time_shifting=0):
	# Generate output file path
	nirspark_export_dir = os.path.join(ds_dir, 'nirspark_export')
	nirspark_export_l = [os.path.join(nirspark_export_dir, f) for f in os.listdir(nirspark_export_dir) if f.endswith('.csv')]
	hbo_normalized_dir = os.path.join(ds_dir, 'hbo')
	hbo_raw_dir = os.path.join(ds_dir, 'hbo_raw')

	os.makedirs(hbo_raw_dir, exist_ok=True)
	os.makedirs(hbo_normalized_dir, exist_ok=True)

	for exported_csv_path in nirspark_export_l:
		sub_label = os.path.basename(exported_csv_path).split('.')[0]
		raw_hbo_path = os.path.join(hbo_raw_dir, f'{sub_label}.csv')

		# Step 2: Rename columns with pandas
		df = pd.read_csv(exported_csv_path, skiprows=4, usecols=range(48), encoding='gbk')


		# Rename columns: replace '-' with '_' and append ' hbo'
		df.columns = [str(col).replace('-', '_') + ' hbo' for col in df.columns]

		# Step 3: Add time column, from 0, increment by 1/sr
		num_rows = df.shape[0]
		time_values = [ time_shifting +i / sr for i in range(num_rows)] 
		df['time'] = time_values

		for col in df.columns:
			if col != "time":  # Skip the "time" column
				# 原始单位为mmol/L*mm
				df[col] *= 5.56  # Multiply by 1000/180=5.56, 即除以光程(30mm)和DPF(6)，单位变成umol/L

		# Save result
		df.to_csv(raw_hbo_path, index=False)

	z_score_normalization(src_dir=hbo_raw_dir, tar_dir=hbo_normalized_dir)

if __name__ == "__main__":
	ds_dir = 'res/trainingcamp-nirspark'
	sr = 11
	time_shifting = 2222.354
	nirspark2csv(ds_dir, sr, time_shifting)
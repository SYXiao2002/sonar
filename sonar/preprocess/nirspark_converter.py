import csv
import os

import pandas as pd


def nirspark2csv(ds_dir, sr, time_shifting=0):
	# Generate output file path
	nirspark_export_dir = os.path.join(ds_dir, 'nirspark_export')
	nirspark_export_l = [os.path.join(nirspark_export_dir, f) for f in os.listdir(nirspark_export_dir) if f.endswith('.csv')]
	out_dir = os.path.join(ds_dir, 'hbo')
	os.makedirs(out_dir, exist_ok=True)

	for raw_csv_path in nirspark_export_l:
		sub_label = os.path.basename(raw_csv_path).split('.')[0]
		new_csv_path = os.path.join(out_dir, f'{sub_label}.csv')

		# Step 1: Remove first 4 rows and keep only first 48 columns
		with open(raw_csv_path, 'r', encoding='utf-8', errors='replace') as infile, \
			open(new_csv_path, 'w', newline='', encoding='utf-8') as outfile:
			
			reader = csv.reader(infile)
			writer = csv.writer(outfile)

			# Skip the first 4 rows
			for _ in range(4):
				next(reader, None)

			# Write remaining rows, only first 48 columns
			for row in reader:
				writer.writerow(row[:48])

		# Step 2: Rename columns with pandas
		df = pd.read_csv(new_csv_path, header=0)

		# Rename columns: replace '-' with '_' and append ' hbo'
		df.columns = [str(col).replace('-', '_') + ' hbo' for col in df.columns]

		# Step 3: Add time column, from 0, increment by 1/sr
		num_rows = df.shape[0]
		time_values = [ time_shifting +i / sr for i in range(num_rows)] 
		df['time'] = time_values

		# Save result
		df.to_csv(new_csv_path, index=False)


if __name__ == "__main__":
	ds_dir = 'res/trainingcamp-nirspark'
	sr = 11
	time_shifting = 2222.354
	nirspark2csv(ds_dir, sr, time_shifting)
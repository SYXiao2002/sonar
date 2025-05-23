import csv
import os

import pandas as pd


def nirspark2csv(nirspark_export, sr, time_shifting=0):
	# Generate output file path
	dirname, basename = os.path.split(nirspark_export)
	filename, ext = os.path.splitext(basename)
	output_file = os.path.join(dirname, f"{filename}_converted{ext}")

	# Step 1: Remove first 4 rows and keep only first 48 columns
	with open(nirspark_export, 'r', encoding='utf-8', errors='replace') as infile, \
		open(output_file, 'w', newline='', encoding='utf-8') as outfile:
		
		reader = csv.reader(infile)
		writer = csv.writer(outfile)

		# Skip the first 4 rows
		for _ in range(4):
			next(reader, None)

		# Write remaining rows, only first 48 columns
		for row in reader:
			writer.writerow(row[:48])

	# Step 2: Rename columns with pandas
	df = pd.read_csv(output_file, header=0)

	# Rename columns: replace '-' with '_' and append ' hbo'
	df.columns = [str(col).replace('-', '_') + ' hbo' for col in df.columns]

	# Step 3: Add time column, from 0, increment by 1/sr
	num_rows = df.shape[0]
	time_values = [ time_shifting +i / sr for i in range(num_rows)] 
	df['time'] = time_values

	# Save result
	df.to_csv(output_file, index=False)


if __name__ == "__main__":
	nirspark_export = "res/20250510wh_test/wh.csv"
	sr = 11
	time_shifting = 0
	nirspark2csv(nirspark_export, sr, time_shifting)
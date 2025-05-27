import mne
import pandas as pd
import os

def get_snirf_metadata(file_path):
	fodler_path = os.path.dirname(file_path)
	csv_path = os.path.join(fodler_path, f"snirf_metadata.csv")
	raw = mne.io.read_raw_snirf(file_path, preload=True)

	# Check if the corresponding csv file exists, if so, return immediately
	if os.path.exists(csv_path):
		print(f'{csv_path} already exists, skipping further extracting.')
		return

	# Read SNIRF file
	ch_names = raw.info['ch_names']
	ch_positions = raw.info['chs']

	# Get first half
	half_len = len(ch_names) // 2
	records = []

	for i in range(half_len):
		ch = ch_names[i]
		base = ch.split(' ')[0] if ' ' in ch else ch
		loc = ch_positions[i]['loc']

		records.append({
			'channel': base,
			'x': loc[0],
			'y': loc[1],
			'z': loc[2],
			'channel_idx': i+1
		})

	# Save to CSV
	df = pd.DataFrame(records)
	df.to_csv(csv_path, index=False)

	print(f"Saved to {csv_path}")

	return df
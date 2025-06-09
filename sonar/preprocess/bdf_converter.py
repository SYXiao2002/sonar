import os
import mne
import numpy as np
import pandas as pd

def bdf2csv(file_path: str, desired_channels: list, csv_path: str,
			first_trigger: str, last_trigger: str):
	"""
	Read BDF file, extract data between two trigger annotations,
	select desired channels, and save to CSV with time.

	Args:
		file_path (str): Path to .bdf file
		desired_channels (list): List of channel names to extract
		csv_path (str): Output CSV path
		first_trigger (str): Annotation description to mark crop start (first occurrence)
		last_trigger (str): Annotation description to mark crop end (last occurrence)
	"""
	# Load raw data
	raw = mne.io.read_raw_bdf(file_path, preload=True)
	annotations = raw.annotations

	# Get all (onset, description) as (float, str)
	events = [(ann['onset'], ann['description'].strip()) for ann in annotations]

	# Get first occurrence of `first_trigger`
	t_start_list = [onset for onset, desc in events if desc == first_trigger]
	if not t_start_list:
		raise ValueError(f"Trigger '{first_trigger}' not found in annotations.")
	t_start = t_start_list[0]

	# Get last occurrence of `last_trigger`
	t_end_list = [onset for onset, desc in events if desc == last_trigger]
	if not t_end_list:
		raise ValueError(f"Trigger '{last_trigger}' not found in annotations.")
	t_end = t_end_list[-1]

	if t_end <= t_start:
		raise ValueError("t_end is not after t_start. Check trigger order.")

	# Crop data
	raw_cropped = raw.copy().crop(tmin=t_start, tmax=t_end)

	# Select channels
	raw_selected = raw_cropped.pick(desired_channels)

	# Get data (shape: n_channels x n_times)
	data, times = raw_selected.get_data(return_times=True)
	data = data.T  # shape: (n_times x n_channels)

	# Save as CSV
	df = pd.DataFrame(data, columns=desired_channels)
	df.insert(0, 'time', times)
	df.to_csv(csv_path, index=False)
	print(f"Saved to: {csv_path}")


def example(ds_dir, desired_channels, out_dir, first_trigger, last_trigger):
	# find all bdf files in the dir/bdf
	bdf_dir = os.path.join(ds_dir, 'bdf')
	bdf_path_l = [os.path.join(bdf_dir, f) for f in os.listdir(bdf_dir) if f.endswith('.bdf')]
	out_dir = os.path.join(ds_dir, out_dir)
	os.makedirs(out_dir, exist_ok=True)

	for bdf_path in bdf_path_l:
		sub_label = os.path.basename(bdf_path).split('.')[0]
		csv_path = os.path.join(out_dir, f'{sub_label}.csv')
		bdf2csv(
		file_path=bdf_path,
		desired_channels=desired_channels,
		csv_path=csv_path,
		first_trigger=str(first_trigger),
		last_trigger=str(last_trigger),
	)


if __name__ == '__main__':
	example(
		ds_dir='res/yuanqu-mne',
		desired_channels=['SPO2', 'Pulse', 'HR'],
		out_dir='heartrate_PPG',
		first_trigger=9,
		last_trigger=9,
	)
import os
from matplotlib import pyplot as plt
import mne
from mne.preprocessing.nirs import (
	optical_density,
	temporal_derivative_distribution_repair,
)
import pandas as pd
from sonar.preprocess.fix_snirf import check_or_create_landmark_labels
from sonar.preprocess.normalization import z_score_normalization
from sonar.preprocess.snirf_metadata import get_snirf_metadata


def read_snirf(file_path):
	"""读取 SNIRF 文件"""
	check_or_create_landmark_labels(file_path)
	raw = mne.io.read_raw_snirf(file_path, preload=True)
	return raw

def get_trigger_times(raw, trigger_id):
	"""获取指定 trigger_id 的第一个和最后一个触发时间点"""
	events, event_dict = mne.events_from_annotations(raw)
	trigger_times = [event[0] / raw.info['sfreq'] for event in events if event[2] == event_dict[str(trigger_id)]]

	if not trigger_times:
		raise ValueError(f"未找到 Trigger {trigger_id}")

	return trigger_times[0], trigger_times[-1]  # 返回第一个和最后一个触发时间点

def crop_data(raw, tmin, tmax):
	"""裁剪数据"""
	raw.crop(tmin, tmax)
	return raw

def convert2hbo(raw_intensity, filter_param_list, debug=False, channel_idx=0):
	"""
	Convert raw intensity to HbO/HbR data.
	If debug=True, plot original and filtered signals with different filter params.

	Parameters:
		raw_intensity: mne.io.Raw
			Raw intensity data.
		filter_param_list: list of (low, high) tuples
			Filter settings to apply. E.g., [(0.01, 0.2), (0.02, 0.15)]
		debug: bool
			Whether to plot raw vs filtered signals.
		channel_idx: int
			Which channel to visualize.
	"""
	# Convert to optical density
	raw_od = optical_density(raw_intensity)

	# Artifact removal
	raw_od = temporal_derivative_distribution_repair(raw_od)

	# Convert to HbO/HbR
	raw_haemo_base = mne.preprocessing.nirs.beer_lambert_law(raw_od)

	if debug:
		# 原始数据
		raw_data_before, times = raw_haemo_base[channel_idx, :]
		raw_data_before = raw_data_before[0]  # shape: (T,)

		plt.figure(figsize=(12, 5))
		plt.plot(times, raw_data_before, label='Raw (Unfiltered)', alpha=0.7)

	# no filter
	if filter_param_list is None:
		return raw_haemo_base

	# 对每一套 filter 参数进行滤波、提取数据、绘图
	filtered_results = []
	for fl, fh, ftl, fth in filter_param_list:
		raw_haemo: mne.io.Raw = raw_haemo_base.copy()
		raw_haemo.filter(l_freq=fl, h_freq=fh, l_trans_bandwidth=ftl, h_trans_bandwidth=fth)

		if debug:
			filtered_data, _ = raw_haemo[channel_idx, :]
			filtered_data = filtered_data[0]
			plt.plot(times, filtered_data, label=f'Filtered {fl-ftl}-{fl}-{fh}-{fh+fth} Hz', alpha=0.7)

		filtered_results.append(raw_haemo)

	if debug:
		plt.title(f'Channel {channel_idx} - Filtering Comparison')
		plt.xlabel('Time (s)')
		plt.ylabel('HbO concentration')
		plt.legend()
		plt.grid(True)
		plt.tight_layout()
		plt.show()

	# 返回最后一套滤波结果（或可返回所有）
	return filtered_results[0]

def process_dataset(ds_dir, time_shifting, first_trigger, last_trigger, filter_param_list, debug, override, thr=30, z_score=True):

	# find all snirf files in the dir/snirf
	hbo_normalized_dir = os.path.join(ds_dir, 'hbo')
	hbo_raw_dir = os.path.join(ds_dir, 'hbo_raw')
	os.makedirs(hbo_normalized_dir, exist_ok=True)
	os.makedirs(hbo_raw_dir, exist_ok=True)

	
	snirf_dir = os.path.join(ds_dir, 'snirf')
	snirf_file_l = [os.path.join(snirf_dir, f) for f in os.listdir(snirf_dir) if f.endswith('.snirf')]
	get_snirf_metadata(snirf_file_l[0])

	for f in snirf_file_l:
		raw = read_snirf(f)

		# Define the roi here!!!
		if first_trigger is not None and last_trigger is not None:
			start_time, _ = get_trigger_times(raw, first_trigger) 
			_, end_time = get_trigger_times(raw, last_trigger)

			# Calculate the expanded cropping range
			tmin_expand = max(0, start_time - thr)
			head_cutting = min(thr, start_time - 0)

			tmax_expand = min(end_time + thr, raw.times[-1])
			tail_cutting = min(thr, raw.times[-1] - end_time)

			print(f'Duration RAW: {raw.times[-1] - raw.times[0]}')
			raw = crop_data(raw, tmin_expand, tmax_expand)
		print(f'Duration Selected: {raw.times[-1] - raw.times[0]}')

		# Convert to HbO/HbR
		hbo = convert2hbo(raw, filter_param_list=filter_param_list, debug=debug)

		if first_trigger is not None and last_trigger is not None:
			# Crop data
			hbo = crop_data(hbo, head_cutting, hbo.times[-1] - tail_cutting)

		# Only keep HbO channels
		hbo.pick_channels([ch for ch in hbo.ch_names if 'hbo' in ch])

		print("Saving data to CSV...")
		df = pd.DataFrame(hbo.get_data().T, columns=hbo.ch_names)
		df["time"] = hbo.times + time_shifting
		
		# Determine the folder and base name of the file
		base_name = os.path.splitext(os.path.basename(f))[0]
		hbo_path = os.path.join(hbo_raw_dir, f'{base_name}.csv')

		for col in df.columns:
			if col != "time":  # Skip the "time" column
				# 原始单位为mol/L
				df[col] *= 1e6  # Multiply by 1e6，单位变成umol/L

		df.to_csv(hbo_path, index=False)

	z_score_normalization(src_dir=hbo_raw_dir, tar_dir=hbo_normalized_dir)

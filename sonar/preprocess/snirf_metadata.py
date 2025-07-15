from typing import Dict, NamedTuple, Tuple
import mne
import numpy as np
import pandas as pd
import os

from sonar.utils.topomap_plot import normalize_positions

class Metadata(NamedTuple):
	pos: Tuple[float, float]
	idx: float

def get_metadata_dict(metadata_path: str) -> Tuple[Dict[str, Metadata], Dict[int, str]]:
	"""
	Load channel metadata from CSV file.

	Returns:
		- name2meta: Dict[channel_name, Metadata]
		- idx2name: Dict[channel_idx, channel_name]
	"""
	df = pd.read_csv(metadata_path)

	name2meta_d = {}
	idx2name_d = {}
	for _, row in df.iterrows():
		ch_name = row['channel_name']
		ch_pos = (row['x'], row['y'])
		ch_idx = int(row['channel_idx'])
		name2meta_d[ch_name] = Metadata(pos=ch_pos, idx=ch_idx)
		idx2name_d[ch_idx] = ch_name

	return name2meta_d, idx2name_d

def normalize_metadata_pos_dict(metadata_dict: Dict[str, Metadata], box_width: float, box_height: float, reverse) -> Dict[str, Metadata]:
	# Step 1: Extract channel names and positions
	ch_names = list(metadata_dict.keys())
	pos_l = np.array([metadata_dict[ch].pos for ch in ch_names])

	# Step 2: Normalize positions
	normed_pos_l = normalize_positions(pos_l, box_width, box_height, reverse=reverse)

	# Step 3: Reconstruct metadata_dict with updated positions
	normed_dict = {
		ch: Metadata(pos=pos, idx=metadata_dict[ch].idx)
		for ch, pos in zip(ch_names, normed_pos_l)
	}

	return normed_dict

def load_idx_remap_dict(metadata_path):
	df = pd.read_csv(metadata_path)
	return dict(zip(df['channel_name'], df['channel_idx']))

def get_snirf_metadata(snirf_path):
	fodler_path = os.path.dirname(snirf_path)
	csv_path = os.path.join(fodler_path, f"snirf_metadata.csv")
	dict_path = os.path.join(fodler_path, f"channel_dict.csv")
	raw = mne.io.read_raw_snirf(snirf_path, preload=True)

	raw_od = mne.preprocessing.nirs.optical_density(raw)
	raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od)
	raw_haemo.pick_channels([ch for ch in raw_haemo.ch_names if 'hbo' in ch])

	# Read SNIRF file
	ch_names = raw_haemo.info['ch_names']
	ch_positions = raw_haemo.info['chs']

	records = []

	# Only load channel dict once if path exists
	ch_dict = None
	if dict_path is not None and os.path.exists(dict_path):
		ch_dict = load_idx_remap_dict(dict_path)

	for i, (ch, pos) in enumerate(zip(ch_names, ch_positions)):
		loc = pos['loc']
		ch = ch.split()[0]
		record = {
			'channel_name': ch,
			'x': loc[0],
			'y': loc[1],
			'z': loc[2],
		}

		if ch_dict is not None and ch in ch_dict:
			record['channel_idx'] = ch_dict[ch]
		else:
			record['channel_idx'] = i + 1

		records.append(record)

	# Save to CSV
	df = pd.DataFrame(records)
	df.to_csv(csv_path, index=False)

	print(f"Saved to {csv_path}")

	return df

if __name__ == '__main__':
	get_snirf_metadata('res/trainingcamp-mne-april/snirf/HC1.snirf')
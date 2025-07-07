# 给定一个通道列表，计算平均; 给定一个sub_l，计算pearson相关性时序，center_win=10s, step=1s


import os
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from sonar.core.dataset_loader import DatasetLoader, get_dataset
from sonar.core.window_selector import WindowSelector
from sonar.preprocess.snirf_metadata import get_metadata_dict


import os
from typing import Optional
import numpy as np
from scipy.stats import pearsonr
from sonar.core.dataset_loader import DatasetLoader


class InterBrainCorrelation():
	def __init__(
		self,
		output_dir: Optional[str],
		dataset: DatasetLoader,
		metadata_path: Optional[str] = None,
		debug: bool = False,
		sub_name_l: Optional[Sequence[str]] = None,
		ch_name_l: Optional[Sequence[str]] = None,
		window_selector: Optional[WindowSelector] = None,
	):
		self.output_dir = output_dir
		self.dataset = dataset
		self.metadata_path = metadata_path
		self.debug = debug
		if sub_name_l is None:
			self.sub_name_l = self.dataset.label_l
		else:
			self.sub_name_l = sub_name_l
		
		if ch_name_l is None:
			self.ch_name_l = self.dataset.ch_l
		else:
			self.ch_name_l = ch_name_l

		if window_selector is None:
			self.window_selector = WindowSelector(
				window_size=10, 
				step=1
			)
		else:
			self.window_selector = window_selector

		os.makedirs(self.output_dir, exist_ok=True)

		self._computed_correlations: Optional[np.ndarray] = None

		self._calculate()
		self._save()

	def _calculate(self):
		"""
		Calculate inter-brain correlation time series.
		1. Average selected channels per subject → 1D signal
		2. Compute Pearson sliding correlation for each subject pair
		3. Average over all pairs
		"""
		ch_idx_l = [self.dataset.ch_l.index(ch_name) for ch_name in self.ch_name_l]
		sub_idx_l = [self.dataset.label_l.index(sub_name) for sub_name in self.sub_name_l]

		# Step 1: Extract and average over channels
		avg_signals = []
		for sub_idx in sub_idx_l:
			sub_data = self.dataset[sub_idx]  # shape: (n_ch, n_time)
			selected_ch = [sub_data[i] for i in ch_idx_l]
			avg_signal = np.mean(selected_ch, axis=0)  # shape: (n_time,)
			avg_signals.append(avg_signal)
		avg_signals = np.array(avg_signals)  # shape: (n_subs, n_time)

		# Step 2–3: Compute all subject-pair sliding correlations
		n_subs = avg_signals.shape[0]
		all_corrs = []
		for i in range(n_subs):
			for j in range(i + 1, n_subs):
				x = avg_signals[i]
				y = avg_signals[j]
				r = self._sliding_pearson(x, y)
				all_corrs.append(r)

		# Step 4: Average over all pairs
		self._computed_correlations = np.mean(all_corrs, axis=0)

	def _sliding_pearson(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
		fs = self.dataset.sr
		win_len = int(self.window_selector.window_size * fs)  # number of samples in window
		step_len = int(self.window_selector.step * fs)       # number of samples to step
		n = len(x)

		corrs = []
		for start in range(0, n - win_len + 1, step_len):
			end = start + win_len
			r, _ = pearsonr(x[start:end], y[start:end])
			corrs.append(r)
		return np.array(corrs)

	def _save(self):
		"""
		Save the computed inter-brain correlation time series to a .tsv file using pandas,
		and save related metadata (e.g. subject/channel info) to a .json file.
		"""
		if self._computed_correlations is None:
			raise ValueError("Correlation results not computed.")

		out_dir = os.path.join(self.output_dir, 'raw_ibc')
		os.makedirs(out_dir, exist_ok=True)

		# Generate a unique filename using timestamp
		from datetime import datetime
		import json

		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		base_filename = f'ibc'

		# Save .tsv correlation data
		save_path = os.path.join(out_dir, base_filename + '.tsv')
		df = pd.DataFrame({
			"time_sec": np.arange(len(self._computed_correlations)) * self.window_selector.step \
				+ self.window_selector.window_size / 2 \
				+ self.dataset['time'][0],
			"correlation": np.clip(self._computed_correlations, 0, None)
		})
		df.to_csv(save_path, sep='\t', index=False)

		# Save metadata to .json
		meta = {
			"sub_name_l": self.sub_name_l,
			"ch_name_l": self.ch_name_l,
			"timestamp": timestamp,
			"time_start": self.dataset['time'][0],
			"window_size_sec": 10,
			"step_size_sec": 1
		}
		meta_path = os.path.join(out_dir, base_filename + '.json')
		with open(meta_path, 'w') as f:
			json.dump(meta, f, indent=4)

		if self.debug:
			print(f"Correlation time series saved to: {save_path}")
			print(f"Metadata saved to: {meta_path}")


def main(ds_dir, load_cache, debug, marker_file, sub_name_l, ch_name_l, use_raw=True):
	ds, _ = get_dataset(ds_dir=os.path.join('res', ds_dir), load_cache=load_cache, marker_file=marker_file, use_raw=use_raw)
	vc = InterBrainCorrelation(
		output_dir = os.path.join('out', ds_dir),
		dataset = ds,
		debug = debug,
		sub_name_l=sub_name_l,
		ch_name_l=ch_name_l
	)

if __name__ == "__main__":
	# ROI
	FRONT_BRAIN_REGION = [22, 27, 24, 25]
	name2meta_d, idx2name_d = get_metadata_dict(metadata_path='res/test/snirf/snirf_metadata.csv')
	front_brain_names = [idx2name_d[idx]+' hbo' for idx in FRONT_BRAIN_REGION]

	# main(ds_dir='test', load_cache=False, debug=True, marker_file=None, sub_name_l=['test-sub2', 'test-sub3'], ch_name_l=front_brain_names)
	main(ds_dir='trainingcamp-homer3', load_cache=False, debug=True, marker_file=None, sub_name_l=['HC1', 'HC3', 'HC5'], ch_name_l=front_brain_names)
	main(ds_dir='trainingcamp-nirspark', load_cache=False, debug=True, marker_file=None, sub_name_l=['HC1', 'HC3', 'HC5'], ch_name_l=front_brain_names)
	main(ds_dir='trainingcamp-mne-april', load_cache=False, debug=True, marker_file=None, sub_name_l=['HC1', 'HC3', 'HC5'], ch_name_l=front_brain_names)
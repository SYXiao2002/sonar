"""
File Name: dataset_loader.py
Author: Yixiao Shen
Date: 2025-05-15
Purpose: Store raw multivariate time series from multiple subjects sharing a common time axis
"""

import csv
import hashlib
import os
import time
from matplotlib import pyplot as plt
from matplotlib.artist import get
import numpy as np
from typing import List, NamedTuple, Sequence, Union
from collections.abc import Sequence as ABCSequence

import pandas as pd
from pyparsing import annotations

from sonar.preprocess.sv_marker import read_annotations

def get_cache_path(dataset_info_list, cache_dir="cache") -> str:
	key_str = "|".join(f"{info.path}:{info.label}" for info in dataset_info_list)
	cache_hash = hashlib.md5(key_str.encode('utf-8')).hexdigest()

	os.makedirs(cache_dir, exist_ok=True)
	return os.path.join(cache_dir, f"dataset_{cache_hash}.npz")

class DatasetInfo(NamedTuple):
	path: str
	label: str

class DatasetLoader:
	def __init__(
		self,
		raw_time: Sequence[float],
		data_l: Sequence[Sequence[Union[np.ndarray, Sequence[float]]]],
		label_l: Sequence[str],
		sr: float,
		ch_l: Sequence[str] = []
	):
		"""
		Container for multivariate time series from multiple subjects sharing a common time axis.

		Parameters:
			raw_time: 1D array-like
				Shared time axis for all time series.
			data_l: Sequence of Sequences of 1D array-like
				Outer sequence represents subjects.
				Inner sequence represents channels for each subject.
			label_list: Sequence of str
				One label per subject (outer sequence).
			sr: float or int
				Sampling rate of the time series.
		"""
		assert isinstance(data_l, ABCSequence) and all(isinstance(subj, ABCSequence) for subj in data_l), \
			"data_l must be a sequence of sequences"
		assert all(len(channel) == len(raw_time) for subject in data_l for channel in subject), \
			"All time series must align with raw_time"
		assert len(data_l) == len(label_l), "Mismatch between number of subjects and labels"

		self.raw_time = raw_time
		self.data_l = data_l
		self.label_l = label_l
		self.sr = sr
		self.ch_l = ch_l

	def __getitem__(self, key):
		if key == "time":
			return self.raw_time
		elif isinstance(key, int):
			return self._get_by_index(key)
		elif isinstance(key, str):
			return self._get_by_label(key)
		else:
			raise TypeError("Key must be int, str or 'time'")
		
	def _get_by_index(self, idx):
		return self.data_l[idx]

	def _get_by_label(self, label):
		idx = self.label_l.index(label)
		return self.data_l[idx]

	def __repr__(self):
		n_subjects = len(self.data_l)
		n_timepoints = len(self.raw_time)
		label_preview = ", ".join(self.label_l[:3])
		if n_subjects > 3:
			label_preview += ", ..."
		return (f"<DatasetLoader with {n_subjects} subjects, "
				f"{n_timepoints} time points, labels: [{label_preview}]>")
	
	def plot_channel(self, subject_idx: int, channel_idx: int):
		"""
		Plot a specific channel vs. time for a given subject.

		Parameters:
			subject_idx: int
				Index of the subject.
			channel_idx: int
				Index of the channel.
		"""
		subject = self.data_l[subject_idx]
		signal = subject[channel_idx]

		channel_name = self.ch_l[channel_idx] if self.ch_l else f'ch{channel_idx + 1}'
		subject_label = self.label_l[subject_idx] if self.label_l else f'sub{subject_idx + 1}'

		plt.figure(figsize=(10, 4))
		plt.plot(self.raw_time, signal)
		plt.title(f'{subject_label} - {channel_name}')
		plt.xlabel('Time (s)')
		plt.ylabel('Signal')
		plt.grid(True)
		plt.tight_layout()
		plt.show()

	def save_cache(self, path: str):
		np.savez_compressed(
			path,
			raw_time=self.raw_time,
			data_l=np.array(self.data_l, dtype=object),  # save list-of-list
			label_l=self.label_l,
			sr=self.sr,
			ch_l=self.ch_l
		)

	@staticmethod
	def from_cache(path: str) -> "DatasetLoader":
		data = np.load(path, allow_pickle=True)
		return DatasetLoader(
			raw_time=data['raw_time'],
			data_l=data['data_l'].tolist(),  # restore as list-of-list
			label_l=data['label_l'].tolist(),
			sr=float(data['sr']),
			ch_l=data['ch_l'].tolist()
		)

	@staticmethod
	def generate_simulated_hbo(subject_label_l: List[str], n_channels: int, sr: int, duration: float, seed: int = 123) -> 'DatasetLoader':
		"""
		Generate a DatasetLoader instance with simulated HbO data.

		:param subject_label_l: List of labels, one per subject
		:param n_channels: Number of channels per subject
		:param sr: Sampling rate (Hz)
		:param duration: Total duration (seconds)
		:param seed: Random seed
		:return: DatasetLoader instance
		"""
		n_timepoints = int(duration * sr)
		time = np.linspace(0, duration, n_timepoints)
		np.random.seed(seed)

		data_l = []
		for _ in subject_label_l:
			# Simulate n_channels x n_timepoints data for each subject
			hbo_data = np.sin(time[None, :] + np.random.randn(n_channels, 1)) * 0.5 \
						+ np.random.randn(n_channels, n_timepoints) * 0.05

			# Split the 2D array into list of 1D arrays, one per channel
			subject_channels = [hbo_data[channel_idx, :] for channel_idx in range(n_channels)]

			data_l.append(subject_channels)

		return DatasetLoader(time, data_l, subject_label_l, sr)

	@staticmethod
	def from_csv_list(dataset_info_list: Sequence[DatasetInfo], load_cache: bool) -> "DatasetLoader":
		"""
		Load multiple CSVs with possibly different lengths.
		Assumes first row is header, and last column is 'time'.
		Truncates all series to minimal common length.
		"""
		cache_path = get_cache_path(dataset_info_list)

		if load_cache and os.path.exists(cache_path):
			print(f"Loading cached dataset from {cache_path}")
			return DatasetLoader.from_cache(cache_path)
	
		all_data = []
		all_time = []
		labels = []
		min_len = None
		ch_l: List[str] = []

		for idx, info in enumerate(dataset_info_list):
			path, label = info.path, info.label
			with open(path, 'r') as f:
				reader = csv.reader(f)
				rows = list(reader)

			header = rows[0]
			ch_names = header[:-1]  # exclude the time column
			if idx == 0:
				ch_l = ch_names  # Use first file's channel names

			data_np = np.array(rows[1:], dtype=float)  # skip header
			time = data_np[:, -1]
			data = data_np[:, :-1].T  # channels x time

			if min_len is None or len(time) < min_len:
				min_len = len(time)

			all_time.append(time)
			all_data.append(data)
			labels.append(label)

		# Truncate to minimal length
		raw_time = all_time[0][:min_len]
		truncated_data = [data[:, :min_len] for data in all_data]
		data_l = [list(map(list, subj)) for subj in truncated_data]

		dt = np.diff(raw_time)
		sr = 1.0 / np.median(dt) if len(dt) > 0 else 0.0

		dataset = DatasetLoader(
			raw_time=raw_time,
			data_l=data_l,
			label_l=labels,
			sr=sr,
			ch_l=ch_l
		)
		dataset.save_cache(cache_path)
		return dataset
	

def get_dataset(ds_dir, load_cache, marker_file=None, use_raw=False):
	if use_raw:
		hbo_dir = os.path.join(ds_dir, 'hbo_raw')
	else:
		hbo_dir = os.path.join(ds_dir, 'hbo')
	if marker_file is None:
		marker_file = os.path.join(ds_dir, 'marker', 'marker.csv')
	else:
		marker_file = os.path.join(marker_file)

	hbo_file_l = [
		os.path.join(hbo_dir, f) for f in os.listdir(hbo_dir) if f.endswith('.csv')
	]

	print(f'hbo_file_l: {hbo_file_l}')

	dataset_template =[
		DatasetInfo(f, os.path.basename(f).split('.')[0]) for f in hbo_file_l
	]

	start_time = time.time()  # Start timing
	dataset = DatasetLoader.from_csv_list(dataset_template, load_cache=load_cache)
	end_time = time.time()  # End timing
	print(f"[INFO] Dataset loaded in {end_time - start_time:.3f} seconds")
	annotations = read_annotations(marker_file)
	return dataset, annotations



def extract_hbo(dataset: DatasetLoader, output_dir: str):
	output_dir = os.path.join(output_dir, 'raw-sv-marker-hbo')
	for sub_idx, sub_label in enumerate(dataset.label_l):
		subject_data = dataset[sub_idx]
		sub_dir = os.path.join(output_dir, sub_label)
		os.makedirs(sub_dir, exist_ok=True)
		time = dataset['time']
		for ch_idx, ch_data in enumerate(subject_data):
			path = os.path.join(sub_dir, f'{ch_idx+1}.csv')
			df = pd.DataFrame({'TIME': time, 'VALUE': ch_data, 'LABEL': f'{sub_label}-ch{ch_idx+1:02d}'})
			df.to_csv(path, index=False)


if __name__ == '__main__':
	ds, _ = get_dataset(ds_dir=os.path.join('res', 'trainingcamp-nirspark'), load_cache=False)
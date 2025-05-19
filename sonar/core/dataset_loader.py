"""
File Name: dataset_loader.py
Author: Yixiao Shen
Date: 2025-05-15
Purpose: Store raw multivariate time series from multiple subjects sharing a common time axis
"""
from typing import List, Sequence, Union
import numpy as np
from collections.abc import Sequence as ABCSequence

class DatasetLoader:
	def __init__(
		self,
		raw_time: Sequence[float],
		data_l: Sequence[Sequence[Union[np.ndarray, Sequence[float]]]],
		label_l: Sequence[str],
		sr: float
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
		self.ts_l = data_l
		self.label_l = label_l
		self.sr = sr

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
		return self.ts_l[idx]

	def _get_by_label(self, label):
		idx = self.label_l.index(label)
		return self.ts_l[idx]

	def __repr__(self):
		n_subjects = len(self.ts_l)
		n_timepoints = len(self.raw_time)
		label_preview = ", ".join(self.label_l[:3])
		if n_subjects > 3:
			label_preview += ", ..."
		return (f"<DatasetLoader with {n_subjects} subjects, "
				f"{n_timepoints} time points, labels: [{label_preview}]>")

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

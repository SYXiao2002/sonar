"""
File Name: tsim_calculator.py
Author: Yixiao Shen
Date: 2025-05-15
Purpose: temporal similarity calculator
"""

import os
from typing import List, Optional, Sequence, Tuple
from matplotlib import pyplot as plt
import numpy as np

from sonar.core.analysis_context import AnalysisContext, SubjectChannel
from sonar.core.dataset_loader import DatasetLoader
from sonar.core.region_selector import RegionSelector
from sonar.core.window_selector import WindowSelector


class CorrelationCalculator(AnalysisContext):
	def __init__(
		self,
		dataset: DatasetLoader,
		window_selector: Optional[WindowSelector] = None,
		region_selector: Optional[RegionSelector] = None
	):
		super().__init__(dataset, window_selector, region_selector)


	def _sliding_window_avg_correlation(self, ts_list, time, window_size, step_size):
		"""
		Use sliding window to calculate average Pearson correlation among all pairs in ts_list.

		:param ts_list: list of 1D numpy arrays, each representing a time series
		:param time: 1D numpy array, time axis shared by all series
		:param window_size: int, number of time points per window
		:param step_size: int, number of time points to move per step
		:return: (sim_times, sim_scores)
		"""
		sim_scores = []
		time_points = []

		num_ts = len(ts_list)

		for i in range(0, len(time) - window_size + 1, step_size):
			# Extract windowed data for all series
			windowed_data = [ts[i:i + window_size] for ts in ts_list]

			# Compute all pairwise correlations
			corrs = []
			for j in range(num_ts):
				for k in range(j + 1, num_ts):
					w1 = windowed_data[j]
					w2 = windowed_data[k]
					if np.std(w1) == 0 or np.std(w2) == 0:
						c = 0.0
					else:
						c = np.corrcoef(w1, w2)[0, 1]
					corrs.append(c)

			# Average over all valid pairs
			score = np.mean(corrs) if corrs else 0.0
			sim_scores.append(score)
			time_points.append(i + window_size // 2)

		sim_times = np.array(time)[time_points]
		sim_scores = np.array(sim_scores)

		return sim_times, sim_scores

	def _global_avg_correlation(self, ts_list: List[np.ndarray]) -> float:
		"""
		Compute global average Pearson correlation among all pairs of time series.

		:param ts_list: list of 1D numpy arrays
		:return: average correlation
		"""
		corrs = []
		num_ts = len(ts_list)
		for i in range(num_ts):
			for j in range(i + 1, num_ts):
				ts1 = ts_list[i]
				ts2 = ts_list[j]
				if np.std(ts1) == 0 or np.std(ts2) == 0:
					c = 0.0
				else:
					c = np.corrcoef(ts1, ts2)[0, 1]
				corrs.append(c)
		return np.mean(corrs)
	
	def _plot(self, 
		  raw_data_l, label_l, raw_times, tsim_times, tsim_scores, 
		  output_path):
		plt.figure(figsize=(12, 8))
		if self.region_selector is None:
			region_selector = RegionSelector(start_sec=raw_times[0], end_sec=raw_times[-1])  # 假设 raw_times 是有序的
		else:
			region_selector = self.region_selector

		# Subplot 1: Raw signals
		plt.subplot(2, 1, 1)
		for ts, label in zip(raw_data_l, label_l):
			plt.plot(raw_times, ts, label=label, alpha=0.7)
		plt.xlim(region_selector.get_xlim_range())
		plt.title(f"Raw Signals {region_selector}")
		plt.grid(True)
		plt.legend(loc='lower right')

		# Subplot 2: Similarity score
		plt.subplot(2, 1, 2)
		plt.axhline(y=0, color='red', linestyle='--', linewidth=2)  # Red horizontal line at y=0
		plt.plot(tsim_times, tsim_scores, color='black', label="Similarity")
		overall_score = np.mean(tsim_scores)
		plt.title(f'Temporal Similarity {self.window_config}: Overall Score  = {overall_score*100:.0f}%')
		plt.grid(True)
		plt.ylim([-1, 1])
		plt.legend(loc='lower right')
		plt.xlim(region_selector.get_xlim_range())
		plt.tight_layout()

		if output_path is not False:
			folder_path = os.path.dirname(output_path)
			if not os.path.exists(folder_path):
				os.makedirs(folder_path)
			plt.savefig(output_path)
		else:
			plt.show()

	def correlation(self, sc_context: Sequence[SubjectChannel], output_path=None) -> Tuple[float, List[float]]:
		"""Compute correlation (mean and per-window) for given time series configuration."""
		sr = self.dataset.sr
		raw_time = self.dataset['time']

		# Apply time restriction if provided
		ts_list = [self.dataset[sc.sub_idx][sc.ch_idx] for sc in sc_context]

		if self.region_selector is not None:
			ts_list, raw_time = self.region_selector.crop_time_series(ts_list, raw_time)

		length = len(raw_time)
		if length == 0:
			raise ValueError("No data in the selected time range")
		
		if self.window_config is None:
			# Use global correlation if no window config is provided
			score = self._global_avg_correlation(ts_list)
			sim_times = raw_time
			sim_scores = [score] * len(raw_time)	# repeat score for consistency

		else:
			# Use sliding window correlation
			window_sec = self.window_config.window_size
			step_sec = self.window_config.step
			window_size = int(window_sec * sr)
			step_size = int(step_sec * sr)

			sim_times, sim_scores = self._sliding_window_avg_correlation(ts_list, raw_time, window_size, step_size)

		if output_path is not None:
			label_l=[f'Ch{sc.ch_idx:02d}-{self.dataset.label_l[sc.sub_idx]}' for sc in sc_context]
			self._plot(
				raw_data_l=ts_list,
				label_l=label_l,
				raw_times=raw_time,
				tsim_times=sim_times,
				tsim_scores=sim_scores,
				output_path=output_path
			)

		return sim_times, sim_scores

	@staticmethod
	def example_demo(show_plot=True):
		# Define subject labels for simulation
		ts_label_l = [
			'Sub01-mne',
			'Sub02-mne',
			'Sub03-mne',
			'Sub01-nirspark',
			'Sub02-nirspark',
			'Sub03-nirspark'
		]

		# Generate simulated HbO data
		dataset = DatasetLoader.generate_simulated_hbo(
			subject_label_l=ts_label_l,
			n_channels=48,
			sr=11,
			duration=100
		)

		# Set time cropping range and sliding window parameters
		region_selector = RegionSelector(center_sec=50, length_sec=10)
		window_selector = WindowSelector(window_size=2, step=1)

		# Create correlation calculator with analysis context
		correlation_calculator = CorrelationCalculator(
			dataset=dataset,
			window_selector=window_selector,
			region_selector=region_selector
		)

		# Select channels from simulated subjects
		sc_context = [
			SubjectChannel(0, 1),
			SubjectChannel(1, 1),
			SubjectChannel(2, 1),
		]

		# Perform correlation analysis without saving to file
		if show_plot:
			correlation_calculator.correlation(
				sc_context=sc_context,
				output_path=False
			)
		else:
			correlation_calculator.correlation(
				sc_context=sc_context,
				output_path=None
			)


if __name__ == '__main__':
	CorrelationCalculator.example_demo()
"""
File Name: trend_topomap.py
Author: Yixiao Shen
Date: 2025-05-19
Purpose: Trend Topomap for individual
"""
import hashlib
import os
import pandas as pd
from tqdm import tqdm
from typing import Dict, Literal, Optional, Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from sonar.core.analysis_context import SubjectChannel
from sonar.core.dataset_loader import DatasetLoader
from sonar.core.region_selector import RegionSelector
from sonar.core.window_selector import WindowSelector
from sonar.preprocess.sv_marker import Annotation
from sonar.utils.trend_detector import TrendDetector


class TrendTopomap():
	def __init__(
		self,
		dataset: DatasetLoader,
		window_selector: WindowSelector,
		mode: Literal['increasing', 'decreasing'] = 'increasing',
		min_duration: float = 5.0,
		region_selector: Optional['RegionSelector'] = None,
		annotations: Optional[Sequence[Annotation]] = None,
		debug: bool = False
	):
		self.dataset = dataset
		self.mode = mode
		self.min_duration = min_duration
		self.annotations = annotations or []
		self.debug = debug
		self.window_selector = window_selector

		if region_selector is None:
			region_selector = RegionSelector(
				start_sec=dataset['time'][0],
				end_sec=dataset['time'][-1]
			)
		self.region_selector = region_selector

		# Save computed results
		self._computed_segments: Dict[int, Sequence[RegionSelector]] = {}

		self._compute_trends()
		self._compute_hidden_calculation()

		print("TrendTopomap initialized.")

	def _compute_trends(self):
		"""
		Detect trend segments and optionally save them.
		"""
		td = TrendDetector(dataset=self.dataset)

		for sub_idx, ts_label in tqdm(enumerate(self.dataset.label_l), total=len(self.dataset.label_l), desc="Subjects"):
			data = np.array(self.dataset[sub_idx])
			n_channels = data.shape[0]

			sub_segments = []
			for ch_idx in tqdm(range(n_channels), desc=f"Detecting trends for {ts_label}", leave=False):
				sc_context = [SubjectChannel(sub_idx, ch_idx)]
				segments = td.detect_trends(
					sc_context,
					mode=self.mode,
					min_duration=self.min_duration
				)
				sub_segments.append(segments)
			self._computed_segments[sub_idx] = sub_segments

	def _compute_hidden_calculation(self, window_selector: Optional[WindowSelector] = None):
		"""
		For each subject and each channel, slide a window (2s, 50%) 
		and count how many computed segments have their midpoints 
		within the window.
		"""
		hidden_results = {}
		if window_selector is not None:
			self.window_selector = window_selector

		for sub_idx, ch_segments in tqdm(self._computed_segments.items(), desc="Subjects"):
			time_array = self.dataset['time']
			start_time = time_array[0]
			end_time = time_array[-1]

			window_size = self.window_selector.window_size
			step_size = self.window_selector.step

			num_windows = int((end_time - start_time - window_size) // step_size) + 1
			sub_hidden = []
			total_counts = []

			for ch_idx, segments in tqdm(
				enumerate(ch_segments),
				total=len(ch_segments),
				desc=f"Channels (sub {sub_idx})",
				leave=False
			):
				channel_hidden = []

				# Precompute and sort midpoints
				midpoints = np.array([(seg.start_sec + seg.end_sec) / 2 for seg in segments])
				midpoints.sort()  # if already sorted, you can skip this line

				# Precompute all window start and end times
				win_starts = start_time + np.arange(num_windows) * step_size
				win_ends = win_starts + window_size

				# Use searchsorted to efficiently count midpoints in each window
				start_indices = np.searchsorted(midpoints, win_starts, side='left')
				end_indices = np.searchsorted(midpoints, win_ends, side='left')

				counts = end_indices - start_indices

				# Build result
				channel_hidden = [
					{'window': (s, e), 'count': int(c)}
					for s, e, c in zip(win_starts, win_ends, counts)
				]

				sub_hidden.append(channel_hidden)


			# after all channels processed
			for w_idx in range(num_windows):
				win_start, win_end = sub_hidden[0][w_idx]['window']
				center = (win_start + win_end) / 2
				count_sum = sum(ch_hidden[w_idx]['count'] for ch_hidden in sub_hidden)
				total_counts.append({'center': center, 'count_sum': count_sum})

			hidden_results[sub_idx] = {
				'per_channel': sub_hidden,
				'total_curve': total_counts
			}

		self._hidden_calculation = hidden_results

	def set_region_selector(self, region_selector: RegionSelector):
		self.region_selector = region_selector

	def save_total_curve_to_csv(self, output_dir):
		"""
		Save only the 'total_curve' part of hidden calculation to a CSV file.
		"""
		records = []

		for sub_idx, sub_data in self._hidden_calculation.items():
			for entry in sub_data.get('total_curve', []):
				records.append({
					'sub_idx': sub_idx,
					'sub_label': self.dataset.label_l[sub_idx],
					'center_time': entry['center'],
					'count_sum': entry['count_sum']
				})

		df = pd.DataFrame(records)
		csv_path = f'total_curve_{self.mode}_{self.min_duration}s.csv'
		csv_path = os.path.join(output_dir, csv_path)
		df.to_csv(csv_path, index=False, encoding='utf-8-sig')


	def plot_trends(self, output_dir):
		"""
		Plot trend segments using existing or loaded computed results.
		"""
		for sub_idx, ts_label in enumerate(self.dataset.label_l):
			data = np.array(self.dataset[sub_idx])
			time = self.dataset['time']
			n_channels, n_times = data.shape

			fig, (ax1, ax2, ax3) = plt.subplots(
				nrows=3, ncols=1,
				figsize=(12, 7),
				gridspec_kw={'height_ratios': [4, 1, 1]},
				sharex=True
			)

			# --- ax1: Trend Segments ---
			ax1.set_title(f"Monotonic Trend Segments: {ts_label}\nmode={self.mode}, min_duration = {self.min_duration}s")
			ax1.set_ylabel(f"Channel ({self.mode})")
			ax1.set_ylim(-0.5, n_channels + 0.5)
			yticks = [i + 0.5 for i in range(n_channels)]
			yticklabels = [f'ch {i + 1}' for i in range(n_channels)]
			ax1.set_yticks([yticks[i] for i in range(0, n_channels, 2)])
			ax1.set_yticklabels([yticklabels[i] for i in range(0, n_channels, 2)])
			for label in ax1.get_yticklabels():
				label.set_fontsize(8)
			ax1.invert_yaxis()
			ax1.grid(True)

			# Draw trend segments
			sub_segments = self._computed_segments.get(sub_idx, [])

			for ch_idx, ch_segments in enumerate(sub_segments):
				valid_centers = []
				marker_sizes = []

				for region in ch_segments:
					start, end = region.get_xlim_range()
					center = (start + end) / 2

					if not (self.region_selector.start_sec <= center <= self.region_selector.end_sec):
						continue

					length = end - start
					marker_size = length / 4

					valid_centers.append(center)
					marker_sizes.append(marker_size)

				if valid_centers:  # only call scatter if there's something to draw
					ax1.scatter(valid_centers, [ch_idx + 0.5] * len(valid_centers),
								s=marker_sizes, color='red', zorder=3)

			# --- ax2: Annotations ---
				ax2.set_ylabel("Event")
				ax2.set_yticks([])
				ax2.set_ylim(0, 1)
				ax2.grid(True)
				xticks = self.region_selector.get_integer_ticks(ideal_num_ticks=10)
				ax2.set_xticks(xticks)
				ax2.set_xticklabels([f"{t:.0f}" for t in xticks], fontsize=8)

				label_colors = {}

				def label_to_color(label):
					# Create a consistent RGB colour from the hash of the label
					h = hashlib.md5(label.encode()).hexdigest()
					r = int(h[0:2], 16) / 255.0
					g = int(h[2:4], 16) / 255.0
					b = int(h[4:6], 16) / 255.0
					return (r, g, b)

				for a in self.annotations:
					if a.label not in label_colors:
						label_colors[a.label] = label_to_color(a.label)

					rect = Rectangle(
						(a.start, 0),
						width=a.duration,
						height=1,
						color=label_colors[a.label],
						alpha=0.5,
						label=a.label
					)
					ax2.add_patch(rect)

				handles, labels = ax2.get_legend_handles_labels()
				by_label = dict(zip(labels, handles))
				ax2.legend(by_label.values(), by_label.keys(), loc='upper right')


			# --- ax3: Hidden Calculation Line Plot ---
			ax3.set_ylabel(f"Intensity\n{self.window_selector}")
			ax3.set_xlabel("Time (s)")
			ax3.grid(True)

			hidden_data = self._hidden_calculation.get(sub_idx, {})
			if hidden_data:
				total_curve = hidden_data.get('total_curve', [])
				if total_curve:
					x_vals = [item['center'] for item in total_curve]
					y_vals = [item['count_sum'] for item in total_curve]
					ax3.plot(x_vals, y_vals, color='blue', linewidth=1.5)

			# --- Shared X limits ---
			ax1.set_xlim(self.region_selector.start_sec, self.region_selector.end_sec)
			ax2.set_xlim(self.region_selector.start_sec, self.region_selector.end_sec)
			ax3.set_xlim(self.region_selector.start_sec, self.region_selector.end_sec)

			# --- Finalize ---
			plt.tight_layout()
			if self.debug:
				plt.show()
			else:
				figure_path = f"{ts_label}_trend_topomap_{self.mode}_{self.min_duration}s_{self.region_selector.start_sec:.0f}_{self.region_selector.end_sec:.0f}.png"
				figure_path = os.path.join(output_dir, figure_path)
				plt.savefig(figure_path, dpi=600)
			plt.close()


"""
File Name: trend_topomap.py
Author: Yixiao Shen
Date: 2025-05-19
Purpose: Trend Topomap for individual
"""
import csv
import hashlib
import os
import pandas as pd
from tqdm import tqdm
from typing import Dict, Literal, Optional, Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from sonar.analysis.intensity_analysis import IntensityAnalyzer
from sonar.core.analysis_context import SubjectChannel
from sonar.core.dataset_loader import DatasetLoader, get_dataset
from sonar.core.region_selector import RegionSelector
from sonar.core.window_selector import WindowSelector
from sonar.preprocess.normalization import normalize_to_range
from sonar.preprocess.sv_marker import Annotation
from sonar.utils.trend_detector import Trend, TrendDetector


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
		self._computed_trends: Dict[int, Sequence[Trend]] = {}

		self._compute_trends()
		self._compute_intensity()

		print("TrendTopomap initialized.")

	def _compute_trends(self):
		"""
		Detect trend segments and optionally save them.
		"""
		td = TrendDetector(dataset=self.dataset)

		for sub_idx, ts_label in tqdm(enumerate(self.dataset.label_l), total=len(self.dataset.label_l), desc="Subjects"):
			data = np.array(self.dataset[sub_idx])
			n_channels = data.shape[0]

			trends_l = []
			for ch_idx in tqdm(range(n_channels), desc=f"Detecting trends for {ts_label}", leave=False):
				sc_context = [SubjectChannel(sub_idx, ch_idx)]
				trends = td.detect_trends(
					sc_context,
					mode=self.mode,
					min_duration=self.min_duration
				)
				trends_l.append(trends)
			self._computed_trends[sub_idx] = trends_l

	def _compute_intensity(self, window_selector: Optional[WindowSelector] = None):
		"""
		For each subject and each channel, slide a window (2s, 50%) 
		and count how many computed segments overlap with each window.
		"""
		intensity_l = {}
		if window_selector is not None:
			self.window_selector = window_selector

		for sub_idx, ch_trends in tqdm(self._computed_trends.items(), desc="Subjects"):
			time_array = self.dataset['time']
			start_time = time_array[0]
			end_time = time_array[-1]

			window_size = self.window_selector.window_size
			step_size = self.window_selector.step

			num_windows = int((end_time - start_time - window_size) // step_size) + 1
			sub_hidden = []
			total_counts = []

			for ch_idx, trends in tqdm(
				enumerate(ch_trends),
				total=len(ch_trends),
				desc=f"Channels (sub {sub_idx})",
				leave=False
			):
				channel_hidden = []

				# Precompute all window start and end times
				win_starts = start_time + np.arange(num_windows) * step_size
				win_ends = win_starts + window_size

				# Count number of segments overlapping with each window
				counts = []
				for win_start, win_end in zip(win_starts, win_ends):
					count = sum(
						not (seg.end_sec <= win_start or seg.start_sec >= win_end)
						for seg in trends
					)
					counts.append(count)

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

			intensity_l[sub_idx] = {
				'per_channel': sub_hidden,
				'intensity_l': total_counts
			}

		self.intensity_l = intensity_l


	def set_region_selector(self, region_selector: RegionSelector):
		self.region_selector = region_selector

	def save_trends_to_csv(self, save_path: str):
		"""
		Save computed trends to a CSV file.
		Each row: label (subject id), ch (channel), start, end
		"""
		if not hasattr(self, '_computed_trends') or not self._computed_trends:
			raise ValueError("No computed trends found. Please run _compute_trends() first.")

		rows = []

		for sub_idx, trends in self._computed_trends.items():
			label = self.dataset.label_l[sub_idx]
			for ch_idx, trend_l in enumerate(trends):
				for t in trend_l:
					rows.append((label, ch_idx+1, t.start_sec, t.end_sec))

		with open(save_path, mode='w', newline='', encoding='utf-8-sig') as f:
			writer = csv.writer(f)
			writer.writerow(['label', 'ch', 'start', 'end'])
			writer.writerows(rows)

		print(f"Saved computed trends to {save_path}")

	def save_intensity_to_csv(self, output_dir):
		"""
		Save the 'intensity' part of hidden calculation:
		- A combined 'all subjects' CSV file
		- One CSV file per subject
		"""
		if not os.path.exists(output_dir): 
			os.makedirs(output_dir)

		all_records = []

		for sub_idx, sub_data in self.intensity_l.items():
			records = []

			for entry in sub_data.get('intensity_l', []):	# use renamed field
				record = {
					'label': self.dataset.label_l[sub_idx],
					'time': entry['center'],
					'value': entry['count_sum']
				}
				records.append(record)
				all_records.append(record)

			# Save per-subject CSV
			df_sub = pd.DataFrame(records)
			sub_label = self.dataset.label_l[sub_idx]
			csv_name = f'intensity_{self.mode}_{sub_label}_{self.min_duration}s.csv'
			csv_path = os.path.join(output_dir, csv_name)
			df_sub.to_csv(csv_path, index=False, encoding='utf-8-sig')

		# Save combined CSV
		df_all = pd.DataFrame(all_records)
		csv_path_all = os.path.join(output_dir, f'intensity_{self.mode}_ALL_{self.min_duration}s.csv')
		df_all.to_csv(csv_path_all, index=False, encoding='utf-8-sig')



	def plot_trends(self, output_dir):
		"""
		Plot trend segments using existing or loaded computed results.
		"""
		if not os.path.exists(output_dir): 
			os.makedirs(output_dir)
		for sub_idx, ts_label in enumerate(self.dataset.label_l):
			data = np.array(self.dataset[sub_idx])
			n_channels, n_times = data.shape

			fig, (ax1, ax2, ax3) = plt.subplots(
				nrows=3,
				ncols=1,
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
			trends: Sequence[Trend] = self._computed_trends.get(sub_idx, [])

			for ch_idx, trend_l in enumerate(trends):
				valid_centers = []
				marker_sizes = []

				for trend in trend_l:
					trend: Trend
					start, end = trend.get_xlim_range()
					center = (start + end) / 2

					# Skip if outside selected region
					if not (self.region_selector.start_sec <= center <= self.region_selector.end_sec):
						continue

					valid_centers.append(center)
					marker_sizes.append(trend.height)

					# Draw horizontal line for the segment
					ax1.hlines(y=ch_idx + 0.5, xmin=start, xmax=end, colors='blue', linewidth=1, zorder=2)


				# Draw scatter points for segment centers
				if valid_centers:
					marker_sizes = normalize_to_range(marker_sizes, min_val=1, max_val=10)
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

			intensity_data = self.intensity_l.get(sub_idx, {})
			if intensity_data:
				intensity = intensity_data.get('intensity_l', [])
				if intensity:
					x_vals = [item['center'] for item in intensity]
					y_vals = [item['count_sum'] for item in intensity]

					# Plot the curve
					thr=30
					ax3.hlines(y=thr, xmin=self.region_selector.start_sec, xmax=self.region_selector.end_sec, label =f'thr={thr}', color='red', linestyle='--', linewidth=1)
					analyzer = IntensityAnalyzer(x_vals, y_vals, smooth_size=30, threshold=thr)
					ax3.plot(x_vals, y_vals, label='original', color='blue', linewidth=1.5)
					ax3.plot(analyzer.times, analyzer.smoothed, label='smoothed', color='black', linewidth=1.5, alpha=0.3)
					ax3.legend()

					for analyzer_seg in analyzer.segments:
						mask = (analyzer.times >= analyzer_seg.start_sec) & (analyzer.times <= analyzer_seg.end_sec)
						ax3.plot(analyzer.times[mask], analyzer.smoothed[mask], 'ro', markersize=2)

					
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

	@staticmethod
	def example():
		dataset, annotations = get_dataset(ds_dir='res/test', load_cache=False)

		window_selector = WindowSelector(window_size=2, step=0.1)
		trend_topomap = TrendTopomap(dataset, window_selector=window_selector, mode='increasing', min_duration=1.6, annotations=annotations, region_selector=None, debug=False)
		trend_topomap.save_intensity_to_csv(output_dir='out/test/intensity_raw')
		trend_topomap.plot_trends(output_dir='out/test/trends_fig')
		trend_topomap.save_trends_to_csv(save_path='out/test/trends_raw.csv')

def save_binary_ts_by_subject(csv_path, output_dir, sample_rate):
	"""
	Convert trend segments to binary time series and save one CSV per subject.
	Each file includes columns: time, ch0, ch1, ...
	:param csv_path: CSV path with columns: label, ch, start, end
	:param output_dir: Folder to save subject-wise CSVs
	:param sample_rate: Sampling rate in Hz
	"""
	df = pd.read_csv(csv_path)

	# 时间轴范围
	start_time = df['start'].min()
	end_time = df['end'].max()
	n_points = int(np.ceil((end_time - start_time) * sample_rate)) + 1
	time_axis = np.linspace(start_time, end_time, n_points)

	# 创建输出文件夹
	os.makedirs(output_dir, exist_ok=True)

	for subject_label, subject_df in df.groupby('label'):
		ch_dict = {}

		for ch, ch_df in subject_df.groupby('ch'):
			binary_array = np.zeros_like(time_axis)
			for _, row in ch_df.iterrows():
				start_idx = int((row['start'] - start_time) * sample_rate)
				end_idx = int((row['end'] - start_time) * sample_rate)
				binary_array[start_idx:end_idx + 1] = 1
			ch_dict[f'ch{ch}'] = binary_array

		# 构建 DataFrame
		df_out = pd.DataFrame({'time': time_axis})
		for ch_col, bin_seq in ch_dict.items():
			df_out[ch_col] = bin_seq

		# 保存
		save_path = os.path.join(output_dir, f"{subject_label}.csv")
		df_out.to_csv(save_path, index=False)
		print(f"Saved: {save_path}")

if __name__ == "__main__":
	TrendTopomap.example()
	save_binary_ts_by_subject('out/test/trends_raw.csv', output_dir='out/test/binery_trends_raw', sample_rate=1.0)

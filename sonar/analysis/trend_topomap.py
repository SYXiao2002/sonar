"""
File Name: trend_topomap.py
Author: Yixiao Shen
Date: 2025-05-19
Purpose: Trend Topomap for individual
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import os
import matplotlib
from matplotlib.pylab import f
import pandas as pd
from tqdm import tqdm
from typing import Dict, Literal, Optional, Sequence

import numpy as np
from matplotlib import pyplot as plt

from sonar.analysis.intensity_analysis import IntensityAnalyzer, Peak
from sonar.core.analysis_context import SubjectChannel
from sonar.core.dataset_loader import DatasetLoader, get_dataset
from sonar.core.region_selector import RegionSelector
from sonar.core.window_selector import WindowSelector
from sonar.preprocess.normalization import normalize_to_range
from sonar.preprocess.sv_marker import Annotation
from sonar.utils.file_helper import clear_folder
from sonar.utils.trend_detector import Trend, TrendDetector

matplotlib.use('Agg')

@dataclass
class Intensity:
	time: float	# Center time of the window
	ch_count: int	# Total trend count in this window


class TrendTopomap():
	def __init__(
		self,
		output_dir: Optional[str],
		dataset: DatasetLoader,
		intensity_window_selector: WindowSelector,
		mode: Literal['increasing', 'decreasing'] = 'increasing',
		min_duration: float = 5.0,
		region_selector: Optional['RegionSelector'] = None,
		annotations: Optional[Sequence[Annotation]] = None,
		debug: bool = False,
		high_intensity_thr: int = 30,
		heartrate_dir: Optional[str] = 'res/heartrate/test'
	):
		self.dataset = dataset
		self.mode = mode
		self.min_duration = min_duration
		self.annotations = annotations if annotations is not None else []
		self.debug = debug
		self.intenisty_window_selector = intensity_window_selector
		self.output_dir = output_dir
		self.thr = high_intensity_thr
		self.heartrate_dir = heartrate_dir

		os.makedirs(self.output_dir, exist_ok=True)

		if region_selector is None:
			region_selector = RegionSelector(
				start_sec=dataset['time'][0],
				end_sec=dataset['time'][-1]
			)
		self.region_selector = region_selector

		# Save computed results
		self._computed_trends: Dict[int, Sequence[Trend]] = {}
		self._computed_trends_binery: Dict[int, Sequence[Sequence]] = {}
		self._computed_intensity: Dict[int, Sequence[Intensity]] = {}
		self._computed_high_intensity: Dict[int, Sequence[IntensityAnalyzer]] = {}
		self._heart_rate: Dict[int, (Sequence[float], Sequence[float])] = {}

		# calculate
		self._calculate()
		self._save()

	def _save(self):
		print("Saving results...")
		self._save_trends_to_csv()
		self._save_intensity_to_csv()
		self._save_trends_binery_to_csv()
		self._save_high_intensity_to_csv()

	def _calculate(self):
		self._compute_trends()
		self._convert_trends_to_binary()
		self._compute_intensity()
		self._extract_high_intensity(thr=self.thr)
		self._compute_channel_status()
		self._get_heart_rate()


	def _get_heart_rate(self):
		for sub_idx, sub_label in enumerate(self.dataset.label_l):
			heart_rate_csv = os.path.join(self.heartrate_dir, f'{sub_label}.csv')
			df = pd.read_csv(heart_rate_csv)

			self._heart_rate[sub_idx] = (df['time'], df['freq'])

	def _compute_trends(self):
		"""
		Detect trend segments for each subject and channel, and optionally save them.
		"""
		trend_detector = TrendDetector(dataset=self.dataset)

		for sub_idx, sub_label in tqdm(
			enumerate(self.dataset.label_l),
			total=len(self.dataset.label_l),
			desc="_compute_trends"
		):
			subject_data = np.array(self.dataset[sub_idx])
			n_channels = subject_data.shape[0]

			subject_trends = []
			for ch_idx in tqdm(
				range(n_channels),
				desc=f"Detecting trends for {sub_label}",
				leave=False
			):
				subject_channel = [SubjectChannel(sub_idx, ch_idx)]
				channel_trends = trend_detector.detect_trends(
					subject_channel,
					mode=self.mode,
					min_duration=self.min_duration
				)
				subject_trends.append(channel_trends)

			self._computed_trends[sub_idx] = subject_trends

	def _save_trends_to_csv(self):
		"""
		Save computed trends to separate CSV files for each subject using pandas.
		Each CSV contains rows: label (subject id), ch (channel), start, end
		"""
		trends_csv_dir = os.path.join(self.output_dir, 'raw_trends')
		os.makedirs(trends_csv_dir, exist_ok=True)

		for sub_idx, sub_trends in self._computed_trends.items():
			sub_label = self.dataset.label_l[sub_idx]

			# Collect all trend rows for this subject
			records = [
				(sub_label, ch_idx + 1, trend.start_sec, trend.end_sec)
				for ch_idx, ch_trends in enumerate(sub_trends)
				for trend in ch_trends
			]

			if records:  # Avoid saving empty files
				df = pd.DataFrame(records, columns=['sub_label', 'ch_idx', 'start_sec', 'end_sec'])
				filename = f"{sub_label}.csv"
				df.to_csv(os.path.join(trends_csv_dir, filename), index=False)

	def _convert_trends_to_binary(self):
		time_array = self.dataset['time']
		n_timepoints = len(time_array)

		for sub_idx, sub_trends in tqdm(self._computed_trends.items(), desc="_convert_trends_to_binary"):
			binary_channels = []

			for ch_trends in tqdm(sub_trends, desc=f"Channels (sub {sub_idx})", leave=False):
				binary_arr = np.zeros(n_timepoints, dtype=np.uint8)

				for trend in ch_trends:
					# Find corresponding index in time array
					start_idx = np.searchsorted(time_array, trend.start_sec, side="left")
					end_idx = np.searchsorted(time_array, trend.end_sec, side="right")
					binary_arr[start_idx:end_idx] = 1

				binary_channels.append(binary_arr)

			self._computed_trends_binery[sub_idx] = binary_channels

			assert all(len(arr) == n_timepoints for arr in binary_channels)

	def _save_trends_binery_to_csv(self):
		binary_csv_dir = os.path.join(self.output_dir, 'raw_trends_binary')
		os.makedirs(binary_csv_dir, exist_ok=True)

		for sub_idx, sub_binary_trends in self._computed_trends_binery.items():
			label = self.dataset.label_l[sub_idx]
			csv_path = os.path.join(binary_csv_dir, f"{label}.csv")

			df = pd.DataFrame(sub_binary_trends).T
			df.columns = [f'ch{i+1}' for i in range(df.shape[1])]

			time_array = self.dataset['time']
			df.insert(0, 'time', time_array)

			df.to_csv(csv_path, index=False)

	def _compute_intensity(self, window_selector: Optional[WindowSelector] = None):
		if window_selector is not None:
			self.intenisty_window_selector = window_selector

		time_array = self.dataset['time']
		dt = time_array[1] - time_array[0]  # sampling interval
		window_size = self.intenisty_window_selector.window_size
		step_size = self.intenisty_window_selector.step

		n_samples = len(time_array)
		win_len = int(window_size / dt)
		step_len = int(step_size / dt)
		num_windows = (n_samples - win_len) // step_len + 1

		for sub_idx, bin_channels in tqdm(self._computed_trends_binery.items(), desc="_compute_intensity"):
			bin_arr = np.array(bin_channels)  # shape: (n_channels, n_samples)
			ch_intensity_list = []
			total_counts = []

			for ch_bin in bin_arr:
				# Apply moving window sum on each channel
				counts = [
					int(np.sum(ch_bin[i * step_len: i * step_len + win_len]))
					for i in range(num_windows)
				]
				ch_intensity_list.append(counts)

			# Aggregate across channels for each window
			for w_idx in range(num_windows):
				win_start_idx = w_idx * step_len
				win_end_idx = win_start_idx + win_len
				center_time = (time_array[win_start_idx] + time_array[win_end_idx - 1]) / 2

				# count how many channels have at least one active point in this window
				ch_count = sum(1 for ch in ch_intensity_list if ch[w_idx] > 0)
				total_counts.append(Intensity(time=center_time, ch_count=ch_count))

			self._computed_intensity[sub_idx] = total_counts

	def _save_intensity_to_csv(self):
		intensity_csv_dir = os.path.join(self.output_dir, 'raw_intensity')
		os.makedirs(intensity_csv_dir, exist_ok=True)


		for sub_idx, sub_intensity in self._computed_intensity.items():
			label = self.dataset.label_l[sub_idx]
			csv_path = os.path.join(intensity_csv_dir, f"{label}.csv")
			df = pd.DataFrame(sub_intensity)
			df.to_csv(csv_path, index=False)

	def _extract_high_intensity(self, thr):
		for sub_idx, sub_intensity in tqdm(self._computed_intensity.items(), desc="_extract_high_intensity"):
			if sub_intensity:
				arr = np.array([(i.time, i.ch_count) for i in sub_intensity])
				x_vals = arr[:, 0]
				y_vals = arr[:, 1]

				intensity_analyzer = IntensityAnalyzer(x_vals, y_vals, smooth_size=30, threshold=thr, max_value=40)

				self._computed_high_intensity[sub_idx] = intensity_analyzer

	def _save_high_intensity_to_csv(self):
		intensity_csv_dir = os.path.join(self.output_dir, 'raw_high_intensity')
		os.makedirs(intensity_csv_dir, exist_ok=True)

		for sub_idx, intensity_analyser in self._computed_high_intensity.items():
			intensity_analyser: IntensityAnalyzer
			label = self.dataset.label_l[sub_idx]
			csv_path = os.path.join(intensity_csv_dir, f"{label}.csv")
			Peak.save_sequence_to_csv(intensity_analyser.segments, csv_path)

	def plot_trends(self, sub_idx=None, out_folder='fig_trends', dpi=300, return_fig=False):
		if sub_idx is None:
			sub_idx_l = range(len(self.dataset.label_l))
			for sub_idx in sub_idx_l:
				self.plot_trends(sub_idx=sub_idx, out_folder=out_folder, dpi=dpi, return_fig=False)
			return

		trends_fig_dir = os.path.join(self.output_dir, out_folder)
		sub_label = self.dataset.label_l[sub_idx]
		os.makedirs(trends_fig_dir, exist_ok=True)
		data = np.array(self.dataset[sub_idx])
		n_channels, n_times = data.shape

		fig, (ax1, ax2, ax3, ax4) = plt.subplots(
			nrows=4,
			ncols=1,
			figsize=(12, 8),
			gridspec_kw={'height_ratios': [4, 1, 1, 1]},
			sharex=True
		)

		# --- ax1: Trend Segments ---
		ax1.set_title(f"Monotonic Trend Segments: {sub_label}\nmode={self.mode}, min_duration = {self.min_duration:.1f}s")
		ax1.set_ylabel(f"Channel ({self.mode})")
		ax1.set_ylim(-0.5, n_channels + 0.5)
		ax1.set_yticks([i + 0.5 for i in range(0, n_channels, 2)])
		ax1.set_yticklabels([f'ch {i + 1}' for i in range(0, n_channels, 2)])
		ax1.invert_yaxis()
		ax1.grid(True)

		# --- optimized trend drawing ---
		sub_trends: Sequence[Trend] = self._computed_trends.get(sub_idx, [])
		lines_xmin, lines_xmax, lines_y = [], [], []
		centers_x, centers_y, center_sizes = [], [], []

		for ch_idx, ch_trends in enumerate(sub_trends):
			for trend in ch_trends:
				if not (self.region_selector.start_sec <= trend.center_sec <= self.region_selector.end_sec):
					continue
				# Accumulate hline data
				lines_xmin.append(trend.start_sec)
				lines_xmax.append(trend.end_sec)
				lines_y.append(ch_idx + 0.5)
				# Accumulate scatter data
				centers_x.append(trend.center_sec)
				centers_y.append(ch_idx + 0.5)
				center_sizes.append(trend.height)

		# Draw all trend segments together
		ax1.hlines(y=lines_y, xmin=lines_xmin, xmax=lines_xmax, colors='blue', linewidth=1, zorder=2)

		if centers_x:
			center_sizes = normalize_to_range(center_sizes, min_val=1, max_val=10)
			ax1.scatter(centers_x, centers_y, s=center_sizes, color='red', zorder=3)

		# --- ax2: Annotations ---
		ax2.set_ylabel("Event")
		ax2.set_yticks([])
		ax2.set_ylim(0, 1)
		ax2.grid(True)
		xticks = self.region_selector.get_integer_ticks(ideal_num_ticks=10)
		ax2.set_xticks(xticks)
		ax2.set_xticklabels([f"{t:.0f}" for t in xticks], fontsize=8)

		label_colors = {}

		@lru_cache(maxsize=None)
		def label_to_color(label):
			h = hashlib.md5(label.encode()).hexdigest()
			return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))

		event_spans = {}
		for a in self.annotations:
			if a.label not in label_colors:
				label_colors[a.label] = label_to_color(a.label)
			if a.label not in event_spans:
				event_spans[a.label] = []
			event_spans[a.label].append((a.start, a.duration))

		for label, spans in event_spans.items():
			ax2.broken_barh(spans, (0, 1), facecolors=label_colors[label], alpha=0.5, label=label)

		ax2.legend(loc='upper right', fontsize=8)

		# --- ax3: Intensity Plot ---
		ax3.set_ylabel(f"Intensity\nwindow={self.intenisty_window_selector.window_size:.1f}s\nstep={self.intenisty_window_selector.step:.1f}s")
		ax3.grid(True)

		intensity = self._computed_intensity.get(sub_idx, {})
		if intensity:
			# Avoid list of tuples → np.array conversion
			x_vals = np.fromiter((i.time for i in intensity), dtype=np.float32)
			y_vals = np.fromiter((i.ch_count for i in intensity), dtype=np.float32)

			analyzer: IntensityAnalyzer = self._computed_high_intensity.get(sub_idx, {})
			thr = analyzer.threshold

			ax3.hlines(
				y=thr,
				xmin=self.region_selector.start_sec,
				xmax=self.region_selector.end_sec,
				label=f'thr={thr}',
				color='red',
				linestyle='--',
				linewidth=1
			)

			ax3.plot(x_vals, y_vals, label='original', color='blue', linewidth=1.5)
			ax3.plot(analyzer.times, analyzer.smoothed, label='smoothed', color='black', linewidth=1.5, alpha=0.3)

			# Vectorized mask creation
			if analyzer.segments:
				segments = np.array([(s.start_sec, s.end_sec) for s in analyzer.segments])
				mask = np.logical_or.reduce([
					(analyzer.times >= start) & (analyzer.times <= end) for start, end in segments
				])
				ax3.plot(analyzer.times[mask], analyzer.smoothed[mask], 'ro', markersize=2)

			ax3.legend()


		# --- ax4: Heart Rate ---
		ax4.set_ylabel("Heart Rate (Hz)")
		ax4.plot(self._heart_rate[sub_idx][0], self._heart_rate[sub_idx][1], color='black', linewidth=1.5)
		ax4.grid(True)
		ax4.set_ylim(1, 1.5)

		# --- Shared x limits ---
		for ax in [ax1, ax2, ax3, ax4]:
			ax.set_xlim(self.region_selector.start_sec, self.region_selector.end_sec)
		ax4.set_xlabel("Time (s)")

		# --- Save ---
		fig.subplots_adjust(hspace=0.3)
		figure_path = f"{sub_label}_trend_topomap_{self.mode}_{self.min_duration}s_{self.region_selector.start_sec:.0f}_{self.region_selector.end_sec:.0f}.png"
		figure_path = os.path.join(trends_fig_dir, figure_path)
		if return_fig:
			return fig, figure_path
		plt.savefig(figure_path, dpi=dpi)
		plt.close()

	def plot_high_intensity(self, out_folder='fig_trends_with_high_intensity'):
		region_selector_tmp = self.region_selector
		save_tasks = []

		for sub_idx, analyzer in tqdm(self._computed_high_intensity.items(), desc="plot_high_intensity"):
			analyzer: IntensityAnalyzer
			region_selector_l: Sequence[RegionSelector] = analyzer.segments
			fig_dir = os.path.join(out_folder, self.dataset.label_l[sub_idx])

			for r in tqdm(region_selector_l, desc=f"Segments for {self.dataset.label_l[sub_idx]}", leave=False):
				padding_sec = 20
				self.region_selector = RegionSelector(
					start_sec=r.start_sec - padding_sec,
					end_sec=r.end_sec + padding_sec
				)
				fig, figure_path = self.plot_trends(sub_idx=sub_idx, out_folder=fig_dir, return_fig=True)
				save_tasks.append((fig, figure_path))

		self.region_selector = region_selector_tmp

		with ThreadPoolExecutor(max_workers=8) as executor:
			for fig, figure_path in save_tasks:
				executor.submit(self._save_fig, fig, figure_path)

	def _save_fig(self, fig, figure_path):
		fig.savefig(figure_path, dpi=300)
		plt.close(fig)

	def set_region_selector(self, region_selector: RegionSelector):
		self.region_selector = region_selector

	def _compute_channel_status(self):
		for sub_idx, analyzer in self._computed_high_intensity.items():
			analyzer: IntensityAnalyzer
			sub_trends: Sequence[Sequence[int]] = self._computed_trends_binery[sub_idx]  # shape: [n_channels][n_timepoints]
			peak_regions: Sequence[Peak] = analyzer.segments
			time_arr = analyzer.times  # shape: [n_timepoints]

			for peak in peak_regions:
				start = peak.start_sec
				end = peak.end_sec

				# 获取事件对应的时间范围 mask
				mask = (time_arr >= start) & (time_arr <= end)

				# 对每个通道检查是否有激活点
				channel_status = []
				for ch_trend in sub_trends:
					# ch_trend[mask] 是该通道在事件时间范围内的二值序列
					active = any(val for val, m in zip(ch_trend, mask) if m)
					channel_status.append(active)

				peak.channel_status = channel_status

	@staticmethod
	def example_run(ds_dir='test'):
		clear_folder(os.path.join('out', ds_dir))
		dataset, annotations = get_dataset(ds_dir=os.path.join('res', ds_dir), load_cache=False)

		window_selector = WindowSelector(window_size=1, step=0.1)

		trend_topomap = TrendTopomap(output_dir=os.path.join('out', ds_dir), 
							   dataset=dataset, intensity_window_selector=window_selector, mode='increasing', 
							   min_duration=1.6, annotations=annotations, region_selector=None, debug=False,
							   high_intensity_thr=30)

		trend_topomap.plot_trends()
		# trend_topomap.plot_high_intensity()

if __name__ == "__main__":
	TrendTopomap.example_run(ds_dir='test')
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

from sonar.analysis.high_density_analysis import HighDensityAnalyzer
from sonar.analysis.density_analysis import DensityAnalyzer, HighDensity
from sonar.core.analysis_context import SubjectChannel
from sonar.core.dataset_loader import DatasetLoader, get_dataset
from sonar.core.region_selector import RegionSelector
from sonar.core.window_selector import WindowSelector
from sonar.preprocess.normalization import normalize_to_range
from sonar.preprocess.sv_marker import Annotation
from sonar.utils.brainregion_plot import annotate_yrange, map_channel_idx_to_y_axis
from sonar.utils.file_helper import clear_folder
from sonar.utils.trend_detector import Trend, TrendDetector

matplotlib.use('Agg')

@dataclass
class Density:
	time: float	# Center time of the window
	ch_count: int	# Total trend count in this window


class TrendTopomap():
	def __init__(
		self,
		output_dir: Optional[str],
		dataset: DatasetLoader,
		density_window_selector: WindowSelector,
		mode: Literal['increasing', 'decreasing'] = 'increasing',
		min_duration: float = 5.0,
		region_selector: Optional['RegionSelector'] = None,
		annotations: Optional[Sequence[Annotation]] = None,
		debug: bool = False,
		high_density_thr: int = 30,
		max_value: Optional[int] = None,
		heartrate_dir: Optional[str] = None
	):
		self.dataset = dataset
		self.mode = mode
		self.min_duration = min_duration
		self.annotations = annotations if annotations is not None else []
		self.debug = debug
		self.intenisty_window_selector = density_window_selector
		self.output_dir = output_dir
		self.thr = high_density_thr
		self.heartrate_dir = heartrate_dir
		self.max_value = max_value

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
		self._computed_density: Dict[int, Sequence[Density]] = {}
		self._computed_high_density: Dict[int, Sequence[DensityAnalyzer]] = {}
		self._heart_rate: Dict[int, (Sequence[float], Sequence[float])] = {}

		# calculate
		self._calculate()
		self._save()

	def _save(self):
		print("Saving results...")
		self._save_trends_to_csv()
		self._save_density_to_csv()
		self._save_trends_binery_to_csv()
		self._save_high_density_to_csv()

	def _calculate(self):
		self._compute_trends()
		self._convert_trends_to_binary()
		self._compute_density()
		self._extract_high_density(thr=self.thr)
		self._compute_channel_status()
		self._get_heart_rate()


	def _get_heart_rate(self):
		if self.heartrate_dir is None:
			return
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

	def _compute_density(self, window_selector: Optional[WindowSelector] = None):
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

		for sub_idx, bin_channels in tqdm(self._computed_trends_binery.items(), desc="_compute_density"):
			bin_arr = np.array(bin_channels)  # shape: (n_channels, n_samples)
			ch_density_list = []
			total_counts = []

			for ch_bin in bin_arr:
				# Apply moving window sum on each channel
				counts = [
					int(np.sum(ch_bin[i * step_len: i * step_len + win_len]))
					for i in range(num_windows)
				]
				ch_density_list.append(counts)

			# Aggregate across channels for each window
			for w_idx in range(num_windows):
				win_start_idx = w_idx * step_len
				win_end_idx = win_start_idx + win_len
				center_time = (time_array[win_start_idx] + time_array[win_end_idx - 1]) / 2

				# count how many channels have at least one active point in this window
				ch_count = sum(1 for ch in ch_density_list if ch[w_idx] > 0)
				total_counts.append(Density(time=center_time, ch_count=ch_count))

			self._computed_density[sub_idx] = total_counts

	def _save_density_to_csv(self):
		density_csv_dir = os.path.join(self.output_dir, 'raw_density')
		os.makedirs(density_csv_dir, exist_ok=True)


		for sub_idx, sub_density in self._computed_density.items():
			label = self.dataset.label_l[sub_idx]
			csv_path = os.path.join(density_csv_dir, f"{label}.csv")
			df = pd.DataFrame(sub_density)
			df.to_csv(csv_path, index=False)

	def _extract_high_density(self, thr):
		for sub_idx, sub_density in tqdm(self._computed_density.items(), desc="_extract_high_density"):
			if sub_density:
				arr = np.array([(i.time, i.ch_count) for i in sub_density])
				x_vals = arr[:, 0]
				y_vals = arr[:, 1]

				density_analyzer = DensityAnalyzer(x_vals, y_vals, smooth_size=30, threshold=thr, max_value=self.max_value)

				self._computed_high_density[sub_idx] = density_analyzer

	def _save_high_density_to_csv(self):
		density_csv_dir = os.path.join(self.output_dir, 'raw_high_density')
		os.makedirs(density_csv_dir, exist_ok=True)

		for sub_idx, density_analyser in self._computed_high_density.items():
			density_analyser: DensityAnalyzer
			label = self.dataset.label_l[sub_idx]
			csv_path = os.path.join(density_csv_dir, f"{label}.csv")
			HighDensity.save_sequence_to_csv(density_analyser.segments, csv_path)

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

		n_subplot =4
		max=10
		ratios = [max, 1, 2, 1]
		if self.heartrate_dir is None:
			n_subplot-=1
			ratios = [max, 2, 1]


		fig, axs = plt.subplots(
			nrows=n_subplot,
			ncols=1,
			figsize=(12, 8),
			gridspec_kw={'height_ratios': ratios, 'hspace': 0.1},
			sharex=True
		)
		fig.subplots_adjust(left=0.1)

		df = pd.read_csv('res/test/snirf_metadata.csv')
		idx2y, mapped_df = map_channel_idx_to_y_axis(df)
		inv_dict = {v: k for k, v in idx2y.items()}

		# --- ax1: Trend Segments ---
		axs[0].set_title(f"Monotonic Trend Segments: {sub_label}\nmode={self.mode}, min_duration = {self.min_duration:.1f}s")
		axs[0].set_yticklabels([])
		axs[0].set_yticks([])
		axs[0].set_ylim(-0.5, n_channels-0.5)

		def pre_frontal_sd_category(ax, width=-0.05):
			annotate_yrange(0, 22, 'Left Hemisphere', offset=width, width=width, text_kwargs={'rotation': 'vertical'}, ax=ax)
			annotate_yrange(25, 47, 'Right Hemisphere', offset=width, width=width, text_kwargs={'rotation': 'vertical'}, ax=ax)
			annotate_yrange(22.5, 24.5, 'Middle', offset=0, width=width*2, text_kwargs={'rotation': 'horizontal', 'fontsize': 9}, ax=ax)

			annotate_yrange(0, 3, 'row\n5', offset=0, width=width, text_kwargs={'rotation': 'vertical', 'fontsize': 9}, ax=ax)
			annotate_yrange(4, 7, 'row\n4', offset=0, width=width, text_kwargs={'rotation': 'vertical', 'fontsize': 9}, ax=ax)
			annotate_yrange(8, 12, 'row\n3', offset=0, width=width, text_kwargs={'rotation': 'vertical', 'fontsize': 9}, ax=ax)
			annotate_yrange(13, 17, 'row\n2', offset=0, width=width, text_kwargs={'rotation': 'vertical', 'fontsize': 9}, ax=ax)
			annotate_yrange(18, 22, 'row\n1', offset=0, width=width, text_kwargs={'rotation': 'vertical', 'fontsize': 9}, ax=ax)
			
			annotate_yrange(44, 47, 'row\n5', offset=0, width=width, text_kwargs={'rotation': 'vertical', 'fontsize': 9}, ax=ax)
			annotate_yrange(40, 43, 'row\n4', offset=0, width=width, text_kwargs={'rotation': 'vertical', 'fontsize': 9}, ax=ax)
			annotate_yrange(35, 39, 'row\n3', offset=0, width=width, text_kwargs={'rotation': 'vertical', 'fontsize': 9}, ax=ax)
			annotate_yrange(30, 34, 'row\n2', offset=0, width=width, text_kwargs={'rotation': 'vertical', 'fontsize': 9}, ax=ax)
			annotate_yrange(25, 29, 'row\n1', offset=0, width=width, text_kwargs={'rotation': 'vertical', 'fontsize': 9}, ax=ax)

			ax.hlines(y=[-0.5, 3.5, 7.5, 12.5, 17.5,  29.5, 34.5, 39.5, 43.5, 47.5], xmin=self.dataset['time'][0], xmax=self.dataset['time'][-1], colors='gray', linewidth=1.5, linestyle='solid', alpha=1, zorder=3)
			ax.hlines(y=[22.5, 24.5], xmin=self.dataset['time'][0], xmax=self.dataset['time'][-1], colors='black', linewidth=1.5, linestyle='solid', alpha=1, zorder=3)

			ax.yaxis.set_ticks_position('right')
			ax.yaxis.set_label_position('right')

			ax.set_yticks([i for i in range(0, n_channels, 1)])
			ax.set_yticklabels([f'ch{inv_dict[i]}' for i in range(0, n_channels, 1)], fontsize=7)
				   
		pre_frontal_sd_category(ax=axs[0])

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
				lines_y.append(idx2y[ch_idx+1])
				# Accumulate scatter data
				centers_x.append(trend.center_sec)
				centers_y.append(idx2y[ch_idx+1])
				center_sizes.append(trend.height)

		# Draw all trend segments together
		axs[0].hlines(y=lines_y, xmin=lines_xmin, xmax=lines_xmax, colors='gray', linewidth=1, zorder=2, alpha=0.4)

		if centers_x:
			center_sizes = normalize_to_range(center_sizes, min_val=0.1, max_val=15)
			axs[0].scatter(centers_x, centers_y, s=center_sizes, color='red', zorder=3)

		# --- ax2: Annotations ---
		axs[1].set_ylabel("Event")
		axs[1].set_yticks([])
		axs[1].set_ylim(0, 1)
		xticks = self.region_selector.get_integer_ticks(ideal_num_ticks=10)
		axs[1].set_xticks(xticks)
		axs[1].set_xticklabels([f"{t:.0f}" for t in xticks], fontsize=8)

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
			axs[1].broken_barh(spans, (0, 1), facecolors=label_colors[label], alpha=0.5, label=label)

		axs[1].legend(loc='upper right', fontsize=8)

		# --- ax3: Density Plot ---
		axs[2].set_ylabel(f"Density\nwindow={self.intenisty_window_selector.window_size:.1f}s\nstep={self.intenisty_window_selector.step:.1f}s")

		density = self._computed_density.get(sub_idx, {})
		if density:
			# Avoid list of tuples → np.array conversion
			x_vals = np.fromiter((i.time for i in density), dtype=np.float32)
			y_vals = np.fromiter((i.ch_count for i in density), dtype=np.float32)

			analyzer: DensityAnalyzer = self._computed_high_density.get(sub_idx, {})
			thr = analyzer.threshold

			axs[2].hlines(
				y=thr,
				xmin=self.region_selector.start_sec,
				xmax=self.region_selector.end_sec,
				label=f'thr={thr}',
				color='red',
				linestyle='--',
				linewidth=1
			)

			if self.max_value is not None:
				axs[2].hlines(
					y=self.max_value,
					xmin=self.region_selector.start_sec,
					xmax=self.region_selector.end_sec,
					label=f'thr-2={self.max_value}',
					color='red',
					linestyle='--',
					linewidth=1
				)

			axs[2].plot(x_vals, y_vals, label='original', color='blue', linewidth=1.5)
			axs[2].plot(analyzer.times, analyzer.smoothed, label='smoothed', color='black', linewidth=1.5, alpha=0.3)

			# Vectorized mask creation
			if analyzer.segments:
				segments = np.array([(s.start_sec, s.end_sec) for s in analyzer.segments])
				mask = np.logical_or.reduce([
					(analyzer.times >= start) & (analyzer.times <= end) for start, end in segments
				])
				axs[2].plot(analyzer.times[mask], analyzer.smoothed[mask], 'ro', markersize=2)

			axs[2].legend(
				bbox_to_anchor=(1.0001, 1),  # (x, y) 坐标，1.02 表示稍微超出右边界，1 表示顶部对齐
				loc="upper left",          # 图例的锚点位置（相对于 bbox_to_anchor）
				borderaxespad=0.           # 图例与轴边界的间距
			)


		# --- ax4: Heart Rate ---
		if self.heartrate_dir is not None:
			axs[3].set_ylabel("HR\n(Hz)")
			axs[3].plot(self._heart_rate[sub_idx][0], self._heart_rate[sub_idx][1], color='black', linewidth=1.5)
			axs[3].set_ylim(1, 1.5)

		# --- Shared x limits ---
		for ax in axs:
			ax.grid(True, axis='x', linestyle='--', color='gray', alpha=0.7)
			ax.set_xlim(self.region_selector.start_sec, self.region_selector.end_sec)
		axs[-1].set_xlabel("Time (s)")

		# --- Save ---
		fig.subplots_adjust(hspace=0.3)
		figure_path = f"{sub_label}_trend_topomap_{self.mode}_{self.min_duration}s_{self.region_selector.start_sec:.0f}_{self.region_selector.end_sec:.0f}.png"
		figure_path = os.path.join(trends_fig_dir, figure_path)
		if return_fig:
			return fig, figure_path
		plt.savefig(figure_path, dpi=dpi)
		plt.close()

	def plot_high_density(self, out_folder='fig_trends_with_high_density'):
		region_selector_tmp = self.region_selector
		save_tasks = []

		for sub_idx, analyzer in tqdm(self._computed_high_density.items(), desc="plot_high_density"):
			analyzer: DensityAnalyzer
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
		if region_selector is None:
			region_selector = RegionSelector(
				start_sec=self.dataset['time'][0],
				end_sec=self.dataset['time'][-1]
			)
		self.region_selector = region_selector

	def _compute_channel_status(self):
		for sub_idx, analyzer in self._computed_high_density.items():
			analyzer: DensityAnalyzer
			sub_trends: Sequence[Sequence[int]] = self._computed_trends_binery[sub_idx]  # shape: [n_channels][n_timepoints]
			peak_regions: Sequence[HighDensity] = analyzer.segments
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

	def permutation_test(self):
		HighDensityAnalyzer(ds_dir=self.output_dir)

	@staticmethod
	def example_run(ds_dir='test'):
		clear_folder(os.path.join('out', ds_dir))
		dataset, annotations = get_dataset(ds_dir=os.path.join('res', ds_dir), load_cache=True)

		window_selector = WindowSelector(window_size=1, step=0.1)

		trend_topomap = TrendTopomap(output_dir=os.path.join('out', ds_dir), 
							   dataset=dataset, density_window_selector=window_selector, mode='increasing', 
							   min_duration=1.6, annotations=annotations, region_selector=None, debug=False,
							   high_density_thr=30)

		trend_topomap.plot_trends()

if __name__ == "__main__":
	TrendTopomap.example_run(ds_dir='test')
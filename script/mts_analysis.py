"""
File Name: mts_analysis.py
Author: Yixiao Shen
Date: 2025-05-28
Purpose: Multi-channel Trend Synchrony
"""

from dataclasses import dataclass
import os

from matplotlib.pylab import f
import numpy as np
import pandas as pd

from sonar.analysis.intensity_analysis import IntensityAnalyzer

from typing import List


@dataclass
class ChannelData:
	label: str
	times: List[float]
	values: List[float]

class PeakAnalysisPipeline:
	def __init__(self, csv_path: str, output_dir: str, smooth_size: int = 50, threshold: int = 30):
		self.csv_path = csv_path
		self.output_dir = output_dir
		self.smooth_size = smooth_size
		self.threshold = threshold
		self.required_columns = {'label', 'time', 'value'}
		self.data: List[ChannelData] = []

	def load_data(self):
		df = pd.read_csv(self.csv_path, encoding='utf-8-sig')

		if not self.required_columns.issubset(df.columns):
			raise ValueError(f"Missing required columns in CSV: {self.required_columns - set(df.columns)}")

		self.data = [
			ChannelData(label=label, times=group['time'].tolist(), values=group['value'].tolist())
			for label, group in df.groupby('label')
		]

	def analyze_and_save(self) -> pd.DataFrame:
		os.makedirs(self.output_dir, exist_ok=True)
		results = []

		for channel_data in self.data:
			analyzer = IntensityAnalyzer(
				times=channel_data.times,
				values=channel_data.values,
				smooth_size=self.smooth_size,
				threshold=self.threshold
			)
			save_path = os.path.join(self.output_dir, f'intensity_peaks_{channel_data.label}.csv')
			result_df = analyzer.save(save_path, label=channel_data.label)
			results.append(result_df)

		all_results = pd.concat(results, ignore_index=True)
		combined_path = os.path.join(self.output_dir, 'intensity_peaks_all.csv')
		all_results.to_csv(combined_path, index=False)
		print(f"[✓] All results saved to: {combined_path}")
		return all_results
	

	@staticmethod
	def example():
		csv_path = 'res/test/intensity_peaks_all.csv'
		output_dir = 'out/intensity_peaks/'

		pipeline = PeakAnalysisPipeline(csv_path, output_dir)
		pipeline.load_data()
		all_df = pipeline.analyze_and_save()

def compute_channel_mts_frequency_from_mts(trend_csv_path, binary_csv_path):
	"""
	基于已知的 MTS 事件段，统计每个通道参与次数
	:param trend_csv_path: trends_raw.csv（MTS事件已提取：每组start+end为一个MTS）
	:param binary_csv_path: 二值趋势CSV（time, ch0, ch1, ...）
	:return: dict, 每个通道对应的参与频率，如 {'ch0': 5, 'ch1': 2, ...}
	"""
	# 读取数据
	trend_df = pd.read_csv(trend_csv_path)
	binary_df = pd.read_csv(binary_csv_path)

	# 只分析一个 label（一个被试）
	label = trend_df['label'].unique()[0]
	trend_df = trend_df[trend_df['label'] == label]

	# 二值时间序列
	time_arr = binary_df['time'].values
	binary_arr = binary_df.drop(columns=['time']).values
	n_channels = binary_arr.shape[1]

	# 按 start+end 分组认为是一个 MTS
	mts_events = trend_df.groupby(['start', 'end'])

	# 初始化通道参与次数
	ch_count = np.zeros(n_channels, dtype=int)

	for (start, end), _ in mts_events:
		# 找出时间段对应的索引
		start_idx = np.searchsorted(time_arr, start, side='left')
		end_idx = np.searchsorted(time_arr, end, side='right')

		segment = binary_arr[start_idx:end_idx]
		if segment.shape[0] == 0:
			continue

		# 某通道是否在该时间段有至少一个1
		active_channels = segment.sum(axis=0) > 0
		ch_count += active_channels.astype(int)

	# 返回为 dict
	return {f'ch{i}': ch_count[i] for i in range(n_channels)}

def plot_channel_freq_heatmap(freq_dict, label='test-sub1'):
	import seaborn as sns
	import matplotlib.pyplot as plt
	df = pd.DataFrame(freq_dict, index=[label])
	sns.heatmap(df, annot=True, cmap='YlOrRd')
	plt.title(f'MTS Participation Frequency for {label}')
	plt.xlabel('Channel')
	plt.ylabel('Subject')
	plt.tight_layout()
	plt.show()

if __name__ == "__main__":
	freq_dict = compute_channel_mts_frequency_from_mts('out/test/trends_raw.csv', 'out/test/binery_trends_raw/test-sub1.csv')
	print(freq_dict)
	plot_channel_freq_heatmap(freq_dict)
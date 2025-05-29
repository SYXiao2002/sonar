import os
import stat
from typing import Optional, Sequence
import numpy as np
import pandas as pd
import scipy
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
import csv

from sonar.core.region_selector import RegionSelector

class Peak(RegionSelector):
	def __init__(self,
		peak_time, peak_value,
		center_sec: Optional[float] = None,
		length_sec: Optional[float] = None,
		start_sec: Optional[float] = None,
		end_sec: Optional[float] = None ,
		):
		super().__init__(center_sec=center_sec, length_sec=length_sec, start_sec=start_sec, end_sec=end_sec)

		self.peak_time = peak_time
		self.peak_value = peak_value

	@staticmethod
	def read_peaks_from_csv(csv_path: str):
		"""
		Read peaks from a CSV file with columns:
		TIME, VALUE, DURATION, LABEL, peak_time, peak_value
		Returns a list of Peak objects.
		"""
		peaks = []
		with open(csv_path, 'r', encoding='utf-8-sig') as f:
			reader = csv.DictReader(f)
			for row in reader:
				peak = Peak(
					start_sec=float(row['TIME']),
					end_sec=float(row['TIME']) + float(row['DURATION']),
					peak_time=float(row['peak_time']),
					peak_value=float(row['peak_value'])
				)
				peaks.append(peak)
		return peaks

class IntensityAnalyzer:
	def __init__(self, times: Sequence[float], values: Sequence[float], smooth_size=50, threshold=30):
		"""
		Initialize the intensity analyzer.

		:param times: Sequence of time points (list or np.array).
		:param values: Corresponding intensity values (list or np.array).
		:param smooth_size: Window size for smoothing the signal.
		:param threshold: Threshold to detect peak segments.
		"""
		self.times = np.array(times)
		self.values = np.array(values)
		self.smooth_size = smooth_size
		self.threshold = threshold
		self.segments: Sequence[Peak] = []
		self.smoothed: Sequence[float] = None

		self._compute()

	def _find_peaks_above_threshold(self, times, values):
		"""
		Find contiguous segments where values exceed the threshold.

		:return: List of Peak(start_sec, end_sec, peak_time, peak_value)
		"""
		mask = values > self.threshold
		labeled_array, num_features = scipy.ndimage.measurements.label(mask)

		segments: Sequence[Peak] = []
		for i in range(1, num_features + 1):
			indices = np.where(labeled_array == i)[0]
			if len(indices) == 0:
				continue

			start_idx = indices[0]
			end_idx = indices[-1]

			# Find peak index within the segment
			segment_values = values[indices]
			relative_peak_idx = np.argmax(segment_values)
			peak_idx = indices[relative_peak_idx]

			peak_time = times[peak_idx]
			peak_value = values[peak_idx]

			peak = Peak(
				peak_time=peak_time,
				peak_value=peak_value,
				start_sec=times[start_idx],
				end_sec=times[end_idx]
			)
			segments.append(peak)

		return segments

	
	def _compute(self):
		"""
		Smooth the values and detect peak segments.

		:return: List of detected peak segments.
		"""
		self.smoothed = uniform_filter1d(self.values, size=self.smooth_size)
		self.segments = self._find_peaks_above_threshold(self.times, self.smoothed)

	def plot(self, show=True, save_path=None):
		"""
		Plot original values, smoothed curve, and detected peak segments.

		:param show: Whether to display the plot.
		:param save_path: Path to save the plot image (optional).
		"""
		if self.smoothed is None:
			raise RuntimeError("Please run analyze() before plotting.")

		plt.figure(figsize=(10, 4))
		plt.plot(self.times, self.values, label='Original', lw=1)
		plt.plot(self.times, self.smoothed, label='Smoothed', lw=2)

		for seg in self.segments:
			mask = (self.times >= seg.start_sec) & (self.times <= seg.end_sec)
			plt.plot(self.times[mask], self.smoothed[mask], 'ro', markersize=4)

		plt.title('Intensity Trend Peaks')
		plt.xlabel('Time (s)')
		plt.ylabel('Intensity')
		plt.legend()
		plt.tight_layout()

		if save_path:
			plt.savefig(save_path)
		if show:
			plt.show()
		else:
			plt.close()


	def save(self, output_path, label) -> pd.DataFrame:
		"""
		Save detected peak segments to CSV in the format:
		TIME, VALUE, DURATION, LABEL, peak_time, peak_value

		:param output_path: Path to output CSV file.
		:param label: Label name to associate with all segments.
		:return: DataFrame of saved segments.
		"""
		if not self.segments:
			raise RuntimeError("No segments to save. Run analyze() first.")

		# 构造 DataFrame
		data = [{
			'TIME': seg.start_sec,
			'VALUE': 1,
			'DURATION': seg.length_sec,
			'LABEL': label,
			'peak_time': seg.peak_time,
			'peak_value': int(seg.peak_value)
		} for seg in self.segments]

		df = pd.DataFrame(data)

		# 保存 CSV
		df.to_csv(output_path, index=False, encoding='utf-8-sig')
		print(f"[✓] Saved peak segments to {output_path}")

		return df
	

	@staticmethod
	def extract_peaks(csv_path, peaks_raw_dir, peaks_fig_dir):
		df = pd.read_csv(csv_path)

		os.makedirs(peaks_raw_dir, exist_ok=True)
		os.makedirs(peaks_fig_dir, exist_ok=True)

		for label, group in df.groupby('label'):
			time = group['time'].tolist()
			value = group['value'].tolist()
			analyzer = IntensityAnalyzer(time, value)
			df = analyzer.save(output_path=os.path.join(peaks_raw_dir, f'{label}.csv'), label=label)
			analyzer.plot(show=False, save_path=os.path.join(peaks_fig_dir, f'{label}.png'))
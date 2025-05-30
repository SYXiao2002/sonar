import os
from typing import Optional, Sequence
import numpy as np
import pandas as pd
import scipy
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt

from sonar.core.region_selector import RegionSelector

class Peak(RegionSelector):
	def __init__(self,
		highest_time, highest_value,
		center_sec: Optional[float] = None,
		length_sec: Optional[float] = None,
		start_sec: Optional[float] = None,
		end_sec: Optional[float] = None ,
		):
		super().__init__(center_sec=center_sec, length_sec=length_sec, start_sec=start_sec, end_sec=end_sec)
		self.highest_time = highest_time
		self.highest_value = highest_value
		self.channel_status: Sequence[bool] = []

	def set_channel_status(self, ch_status: Sequence[bool]):
		self.channel_status = ch_status


	def __repr__(self):
		active_channels = [i for i, active in enumerate(self.channel_status) if active]
		return (f"<Peak start={self.start_sec:.2f}s, end={self.end_sec:.2f}s, "
				f"peak={self.highest_time:.2f}s @ {self.highest_value}, "
				f"active_ch={active_channels}>")

	@classmethod
	def save_sequence_to_csv(cls, peaks: Sequence["Peak"], path: str):
		if not peaks:
			raise ValueError("Empty peak list.")

		num_channels = len(peaks[0].channel_status)
		channel_names = [f"ch{i+1}" for i in range(num_channels)]

		data = []
		for seg in peaks:
			row = {
				'start_sec': seg.start_sec,
				'end_sec': seg.end_sec,
				'length_sec': seg.length_sec,
				'center_sec': seg.center_sec,
				'highest_time': seg.highest_time,
				'highest_value': seg.highest_value
			}
			row.update({ch: int(status) for ch, status in zip(channel_names, seg.channel_status)})
			data.append(row)

		df = pd.DataFrame(data)
		df.to_csv(path, index=False)

	@classmethod
	def load_sequence_from_csv(cls, path: str) -> Sequence["Peak"]:
		df = pd.read_csv(path)
		channel_cols = [col for col in df.columns if col.startswith("ch")]

		peaks = []
		for _, row in df.iterrows():
			peak = cls(
				highest_time=row['highest_time'],
				highest_value=row['highest_value'],
				center_sec=row['center_sec'],
				length_sec=row['length_sec'],
				start_sec=row['start_sec'],
				end_sec=row['end_sec']
			)
			ch_status = [bool(row[ch]) for ch in channel_cols]
			peak.set_channel_status(ch_status)
			peaks.append(peak)

		return peaks


class IntensityAnalyzer:
	def __init__(self, times: Sequence[float], values: Sequence[float], smooth_size=50, threshold=30, max_value=40):
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
		self.max_value = max_value
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
				highest_time=peak_time,
				highest_value=peak_value,
				start_sec=times[start_idx],
				end_sec=times[end_idx]
			)
			if peak_value < self.max_value:
				continue
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

	@staticmethod
	def extract_peaks(csv_path, peaks_raw_dir, peaks_fig_dir):
		df = pd.read_csv(csv_path)

		os.makedirs(peaks_raw_dir, exist_ok=True)
		os.makedirs(peaks_fig_dir, exist_ok=True)

		for label, group in df.groupby('label'):
			time = group['time'].tolist()
			value = group['value'].tolist()
			analyzer = IntensityAnalyzer(time, value, smooth_size=50, threshold=30)
			df = analyzer.save(output_path=os.path.join(peaks_raw_dir, f'{label}.csv'), label=label)
			analyzer.plot(show=False, save_path=os.path.join(peaks_fig_dir, f'{label}.png'))
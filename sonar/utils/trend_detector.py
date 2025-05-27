"""
File Name: trend_detector.py
Author: Yixiao Shen
Date: 2025-05-19
Purpose: 
"""

import matplotlib.pyplot as plt
from typing import Literal, Optional, Sequence

import numpy as np

from sonar.core.analysis_context import AnalysisContext, SubjectChannel
from sonar.core.dataset_loader import DatasetLoader
from sonar.core.region_selector import RegionSelector
from sonar.core.window_selector import WindowSelector


class TrendDetector(AnalysisContext):
	def __init__(
		self,
		dataset: DatasetLoader,
		window_selector: Optional[WindowSelector] = None,
		region_selector: Optional[RegionSelector] = None
	):
		super().__init__(dataset, window_selector, region_selector)

	def _find_monotonic_segments(
		self,
		data: Sequence[float],
		time: Sequence[float],
		min_duration: float = 3.0,
		mode: Literal['increasing', 'decreasing'] = 'increasing',
	) -> Sequence[RegionSelector]:
		if len(data) != len(time):
			raise ValueError("data and time must have the same length")

		sign = 1 if mode == "increasing" else -1
		delta = np.diff(data)
		delta_sign = sign * delta > 0  # Boolean array where monotonicity is satisfied

		# Pad to match time length (delta is len-1)
		delta_sign = np.concatenate([[False], delta_sign])

		segments = []
		start_idx = None

		for i, is_mono in enumerate(delta_sign):
			if is_mono and start_idx is None:
				start_idx = i
			elif not is_mono and start_idx is not None:
				start_time = time[start_idx]
				end_time = time[i - 1]
				if end_time - start_time >= min_duration:
					segments.append(RegionSelector(start_sec=start_time, end_sec=end_time))
				start_idx = None

		# Handle last segment
		if start_idx is not None:
			start_time = time[start_idx]
			end_time = time[-1]
			if end_time - start_time >= min_duration:
				segments.append(RegionSelector(start_sec=start_time, end_sec=end_time))

		return segments

	def detect_trends(
		self,
		sc_context: Sequence[SubjectChannel],
		mode: Literal['increasing', 'decreasing'] = 'increasing',
		min_duration: float = 1.0
	) -> Sequence[RegionSelector]:
		raw_time = self.dataset['time']
		data = self.dataset[sc_context[0].sub_idx][sc_context[0].ch_idx]

		if self.region_selector is not None:
			cropped_data, cropped_time = self.region_selector.crop_time_series([data], raw_time)
			data = cropped_data[0]
			raw_time = cropped_time

		segments = self._find_monotonic_segments(data, raw_time, min_duration=min_duration, mode=mode)

		return segments

	def plot_trends(
		self,
		sc_context: Sequence[SubjectChannel],
		mode: str = "increasing",
		min_duration: float = 1.0
	):
		raw_time = self.dataset['time']
		data = self.dataset[sc_context[0].sub_idx][sc_context[0].ch_idx]

		# if self.region_selector is not None:
		# 	cropped_data, cropped_time = self.region_selector.crop_time_series([data], raw_time)
		# 	data = cropped_data[0]
		# 	raw_time = cropped_time

		segments = self._find_monotonic_segments(data, raw_time, min_duration=min_duration, mode=mode)

		plt.figure(figsize=(12, 6))
		plt.plot(raw_time, data, label='Data')

		for region in segments:
			start, end = region.get_xlim_range()
			plt.axvspan(start, end, color='orange' if mode == 'increasing' else 'blue', alpha=0.3)

		plt.xlabel('Time (s)')
		plt.ylabel('Signal Value')
		plt.title(f"Detected {mode} segments (min duration {min_duration}s)")
		plt.legend()
		plt.show()

	@staticmethod
	def example_demo1():
		import numpy as np

		# 模拟时间和数据：先递增后递减，再递增
		time = np.linspace(0, 30, 300)
		data = np.piecewise(
			time,
			[time < 10, (time >= 10) & (time < 20), time >= 20],
			[lambda t: t, lambda t: 20 - t, lambda t: t - 20 + 10]
		)

		# 构造模拟dataset结构，假设1个subject，1个channel
		dataset = {
			'time': time,
			0: [data]  # dataset[subject][channel] = data
		}
		region_selector = RegionSelector(center_sec=5, length_sec=2)

		# 构造SubjectChannel列表
		sc_context = [SubjectChannel(0, 0)]

		# 初始化TrendDetector
		td = TrendDetector(dataset=dataset, region_selector=region_selector)

		# 检测递增趋势段
		segments = td.detect_trends(sc_context, mode='increasing', min_duration=2)
		print("Detected increasing segments:", segments)

		# 可视化
		td.plot_trends(sc_context, mode='increasing', min_duration=2)

	@staticmethod
	def example_demo2():
		import numpy as np

		dataset = get_test_dataset(load_cache=False)

		region_selector = RegionSelector(center_sec=5, length_sec=2)

		# 构造SubjectChannel列表
		sc_context = [SubjectChannel(0, 0)]

		# 初始化TrendDetector
		td = TrendDetector(dataset=dataset, region_selector=region_selector)

		# 检测递增趋势段
		segments = td.detect_trends(sc_context, mode='increasing', min_duration=0.5)
		print("Detected increasing segments:", segments)

		# 可视化
		td.plot_trends(sc_context, mode='increasing', min_duration=0.5)


if __name__ == "__main__":
	TrendDetector.example_demo1()
	TrendDetector.example_demo2()

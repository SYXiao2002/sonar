"""
File Name: region_selector.py
Author: Yixiao Shen
Date: 2025-05-16
Purpose: Define temporal region selection and cropping utilities for time series analysis.
"""

import numpy as np
from typing import Sequence, Tuple, Optional, Union

import pandas as pd


class RegionSelector:
	def __init__(
		self,
		*,
		center_sec: Optional[float] = None,
		length_sec: Optional[float] = None,
		start_sec: Optional[float] = None,
		end_sec: Optional[float] = None
	):
		"""
		Define a time region using one of the following valid parameter combinations:
			- center_sec + length_sec
			- start_sec + end_sec
			- start_sec + length_sec

		All values are in seconds.
		"""
		self.center_sec = center_sec
		self.length_sec = length_sec
		self.start_sec = start_sec
		self.end_sec = end_sec

		self._compute_xlim()

	def _compute_xlim(self):
		"""Compute derived values from provided input parameters."""
		if self.center_sec is not None and self.length_sec is not None:
			self.start_sec = self.center_sec - self.length_sec / 2
			self.end_sec = self.center_sec + self.length_sec / 2
		elif self.start_sec is not None and self.end_sec is not None:
			self.center_sec = (self.start_sec + self.end_sec) / 2
			self.length_sec = self.end_sec - self.start_sec
		elif self.start_sec is not None and self.length_sec is not None:
			self.end_sec = self.start_sec + self.length_sec
			self.center_sec = (self.start_sec + self.end_sec) / 2
		else:
			raise ValueError(
				"Invalid region specification. Provide one of:\n"
				"- (center_sec & length_sec)\n"
				"- (start_sec & end_sec)\n"
				"- (start_sec & length_sec)"
			)

		self.xlim_range = (self.start_sec, self.end_sec)

	def get_xlim_range(self) -> Tuple[float, float]:
		"""Return (start_sec, end_sec)."""
		return self.xlim_range

	def __repr__(self) -> str:
		return (
			f"[start={int(self.start_sec)}s, end={int(self.end_sec)}s]"
		)
	
	def crop_time_series(self, 
					  ts_l: Sequence[Union[np.ndarray, Sequence[float]]], 
					  time: Sequence[float]) -> Tuple[Sequence[Union[np.ndarray, Sequence[float]]], Sequence[float]]:
		start_idx = np.searchsorted(time, self.start_sec)
		end_idx = np.searchsorted(time, self.end_sec)
		
		cropped_ts_list = [ts[start_idx:end_idx] for ts in ts_l]
		cropped_time = time[start_idx:end_idx]

		return cropped_ts_list, cropped_time
	

	def get_integer_ticks(self, ideal_num_ticks: int = 5) -> list[int]:
		"""
		Generate a list of approximately `ideal_num_ticks` evenly spaced integer ticks
		between self.start_sec and self.end_sec.

		Args:
			ideal_num_ticks (int): Desired number of ticks (default: 5)

		Returns:
			List[int]: List of integer ticks
		"""
		start = int(np.floor(self.start_sec))
		end = int(np.ceil(self.end_sec))
		total_duration = end - start

		if total_duration <= 0:
			return [start]  # fallback: invalid range

		# Compute raw step
		raw_step = max(1, total_duration // (ideal_num_ticks - 1))

		# Round step to a nice number: 1, 2, 5, 10, etc.
		def round_step(s):
			if s <= 1:
				return 1
			elif s <= 2:
				return 2
			elif s <= 5:
				return 5
			elif s <= 10:
				return 10
			else:
				return int(round(s / 10)) * 10

		step = round_step(raw_step)
		ticks = list(range(start, end + 1, step))

		# Ensure the last tick covers the end
		if ticks[-1] < end:
			ticks.append(ticks[-1] + step)

		return ticks
	
	@staticmethod
	def load_from_csv(csv_path): 
		df = pd.read_csv(csv_path)
		region_l = []
		for _, row in df.iterrows():
			region = RegionSelector(
				length_sec=row['length_sec'],
				start_sec=row['start_sec'],
			)
			region_l.append(region)
		return region_l

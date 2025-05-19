"""
File Name: sliding_window.py
Author: Yixiao Shen
Date: 2025-05-15
Purpose: Define sliding window configuration
"""

class WindowSelector:
	def __init__(self, window_size: int, step: int = None, overlap_rate: float = None):
		"""
		Window configuration for sliding window correlation.

		Parameters:
			window_size: int
				Size of the sliding window (must be positive).
			step: int, optional
				Step size between windows. Must be <= window_size if provided.
			overlap_rate: float, optional
				Proportion of overlap between consecutive windows (0 <= r < 1).

		Exactly one of step or overlap_rate must be provided.
		"""
		if window_size <= 0:
			raise ValueError("window_size must be a positive integer")

		if (step is None) == (overlap_rate is None):
			raise ValueError("Exactly one of step or overlap_rate must be provided")

		self.window_size = window_size

		if step is not None:
			if step <= 0 or step > window_size:
				raise ValueError("step must be in the range (0, window_size]")
			self.step = step
			self.overlap_rate = 1 - step / window_size
		else:
			if not (0 <= overlap_rate < 1):
				raise ValueError("overlap_rate must be in the range [0, 1)")
			self.overlap_rate = overlap_rate
			self.step = max(int(window_size * (1 - overlap_rate)), 1)  # Ensure positive step
			if self.step <= 0:
				raise ValueError("calculated step must be positive")

	def __str__(self) -> str:
		"""
		Return a readable string representation of the configuration.
		"""
		return f"[window_size={self.window_size:.1f}s, step={self.step:.1f}s]"

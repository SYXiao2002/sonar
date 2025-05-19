from dataclasses import dataclass
from typing import Optional

from sonar.core.dataset_loader import DatasetLoader
from sonar.core.region_selector import RegionSelector
from sonar.core.window_selector import WindowSelector

@dataclass(frozen=True)
class SubjectChannel:
	sub_idx: int
	ch_idx: int


class AnalysisContext:
	def __init__(
		self,
		dataset: DatasetLoader,
		window_config: Optional[WindowSelector] = None,
		region_selector: Optional[RegionSelector] = None
	):
		"""
		Container for all analysis-level configurations, including:
			- time series data (dataset)
			- optional time region selection (xlim_range)
			- optional sliding window parameters (window_config)
			- subject-channel selection (subject_ch_l, to be set separately)

		Parameters:
			dataset: DatasetLoader
				The time series data container.
			window_config: Optional[WindowSelector]
				Sliding window configuration.
			region_selector: Optional[RegionSelector]
				Time cropping range (start_sec, end_sec).
		"""
		self.dataset = dataset
		self.window_config = window_config
		self.region_selector = region_selector
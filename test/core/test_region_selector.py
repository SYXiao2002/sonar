import numpy as np
import pytest
from sonar.core.region_selector import RegionSelector


def test_init_center_length():
	selector = RegionSelector(center_sec=5.0, length_sec=4.0)
	assert selector.get_xlim_range() == (3.0, 7.0)
	assert selector.center_sec == 5.0
	assert selector.length_sec == 4.0


def test_init_start_end():
	selector = RegionSelector(start_sec=2.0, end_sec=6.0)
	assert selector.get_xlim_range() == (2.0, 6.0)
	assert selector.center_sec == 4.0
	assert selector.length_sec == 4.0


def test_init_start_length():
	selector = RegionSelector(start_sec=3.0, length_sec=6.0)
	assert selector.get_xlim_range() == (3.0, 9.0)
	assert selector.center_sec == 6.0
	assert selector.end_sec == 9.0


def test_invalid_config():
	with pytest.raises(ValueError):
		_ = RegionSelector(center_sec=5.0)  # Missing length


def test_crop_time_series_basic():
	time = np.linspace(0, 10, 11)  # [0, 1, ..., 10]
	ts1 = np.arange(11)           # [0, 1, ..., 10]
	ts2 = np.arange(11, 22)       # [11, 12, ..., 21]

	selector = RegionSelector(start_sec=3.0, end_sec=7.0)
	cropped_ts, cropped_time = selector.crop_time_series([ts1, ts2], time)

	# Check cropped time range
	assert list(cropped_time) == [3.0, 4.0, 5.0, 6.0]
	# Check corresponding values
	assert list(cropped_ts[0]) == [3, 4, 5, 6]
	assert list(cropped_ts[1]) == [14, 15, 16, 17]
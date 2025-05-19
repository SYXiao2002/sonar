import pytest

from sonar.core.window_selector import WindowSelector

def test_valid_with_step():
	ws = WindowSelector(window_size=10, step=5)
	assert ws.window_size == 10
	assert ws.step == 5
	assert ws.overlap_rate == 1 - 5 / 10


def test_valid_with_overlap_rate():
	ws = WindowSelector(window_size=20, overlap_rate=0.25)
	assert ws.window_size == 20
	assert ws.overlap_rate == 0.25
	expected_step = max(int(20 * (1 - 0.25)), 1)
	assert ws.step == expected_step


def test_window_size_invalid():
	with pytest.raises(ValueError, match="window_size must be a positive integer"):
		WindowSelector(window_size=0, step=1)


def test_step_and_overlap_rate_both_none():
	with pytest.raises(ValueError, match="Exactly one of step or overlap_rate must be provided"):
		WindowSelector(window_size=10)


def test_step_and_overlap_rate_both_provided():
	with pytest.raises(ValueError, match="Exactly one of step or overlap_rate must be provided"):
		WindowSelector(window_size=10, step=3, overlap_rate=0.1)


def test_step_invalid_too_large():
	with pytest.raises(ValueError, match="step must be in the range"):
		WindowSelector(window_size=10, step=11)


def test_step_invalid_non_positive():
	with pytest.raises(ValueError, match="step must be in the range"):
		WindowSelector(window_size=10, step=0)


def test_overlap_rate_invalid_negative():
	with pytest.raises(ValueError, match="overlap_rate must be in the range"):
		WindowSelector(window_size=10, overlap_rate=-0.1)


def test_overlap_rate_invalid_equal_one():
	with pytest.raises(ValueError, match="overlap_rate must be in the range"):
		WindowSelector(window_size=10, overlap_rate=1.0)

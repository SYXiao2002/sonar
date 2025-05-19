import numpy as np
import pytest

from sonar.core.analysis_context import SubjectChannel
from sonar.core.region_selector import RegionSelector
from sonar.core.window_selector import WindowSelector
from sonar.core.dataset_loader import DatasetLoader
from sonar.utils.tsim_calculator import CorrelationCalculator


@pytest.fixture
def calculator_fixture():
	ts_label_l = ['Sub01', 'Sub02', 'Sub03']
	dataset = DatasetLoader.generate_simulated_hbo(
		subject_label_l=ts_label_l,
		n_channels=48,
		sr=10,
		duration=30
	)
	region_selector = RegionSelector(center_sec=15, length_sec=10)
	window_selector = WindowSelector(window_size=2, step=1)

	calculator = CorrelationCalculator(
		dataset=dataset,
		window_selector=window_selector,
		region_selector=region_selector
	)
	sc_context = [
		SubjectChannel(0, 0),
		SubjectChannel(1, 0),
		SubjectChannel(2, 0)
	]

	return calculator, sc_context


def test_correlation_output_format(calculator_fixture):
	calculator, sc_context = calculator_fixture
	sim_times, sim_scores = calculator.correlation(sc_context=sc_context, output_path=None)

	assert isinstance(sim_times, np.ndarray)
	assert isinstance(sim_scores, np.ndarray)
	assert sim_times.shape == sim_scores.shape
	assert sim_times.ndim == 1
	assert sim_scores.ndim == 1


def test_region_selector_none():
	ts_label_l = ['S1', 'S2']
	dataset = DatasetLoader.generate_simulated_hbo(
		subject_label_l=ts_label_l,
		n_channels=10,
		sr=5,
		duration=10
	)

	calculator = CorrelationCalculator(
		dataset=dataset,
		window_selector=WindowSelector(window_size=2, step=1),
		region_selector=None
	)
	sc_context = [
		SubjectChannel(0, 0),
		SubjectChannel(1, 0)
	]

	# Should still work without region selector
	sim_times, sim_scores = calculator.correlation(sc_context, output_path=None)
	assert sim_times.shape == sim_scores.shape


def test_window_selector_none():
	ts_label_l = ['A', 'B', 'C']
	dataset = DatasetLoader.generate_simulated_hbo(
		subject_label_l=ts_label_l,
		n_channels=10,
		sr=10,
		duration=5
	)

	calculator = CorrelationCalculator(
		dataset=dataset,
		window_selector=None,  # global correlation mode
		region_selector=None
	)
	sc_context = [
		SubjectChannel(0, 0),
		SubjectChannel(1, 0),
		SubjectChannel(2, 0)
	]

	sim_times, sim_scores = calculator.correlation(sc_context, output_path=None)

	assert np.allclose(sim_scores, sim_scores[0])  # All values should be same in global mode


def test_empty_crop_raises():
	ts_label_l = ['Test']
	dataset = DatasetLoader.generate_simulated_hbo(
		subject_label_l=ts_label_l,
		n_channels=1,
		sr=10,
		duration=5
	)

	region_selector = RegionSelector(start_sec=10.0, end_sec=20.0)  # Outside of data range
	calculator = CorrelationCalculator(
		dataset=dataset,
		window_selector=WindowSelector(window_size=2, step=1),
		region_selector=region_selector
	)
	sc_context = [SubjectChannel(0, 0)]

	with pytest.raises(ValueError, match="No data in the selected time range"):
		calculator.correlation(sc_context, output_path=None)


def test_plot_disabled(monkeypatch):
	ts_label_l = ['X1', 'X2']
	dataset = DatasetLoader.generate_simulated_hbo(
		subject_label_l=ts_label_l,
		n_channels=5,
		sr=10,
		duration=5
	)

	calculator = CorrelationCalculator(
		dataset=dataset,
		window_selector=WindowSelector(window_size=2, step=1),
		region_selector=None
	)
	sc_context = [SubjectChannel(0, 0), SubjectChannel(1, 0)]

	# Mock _plot to ensure it's not called when output_path is None
	called = {'value': False}
	def mock_plot(*args, **kwargs):
		called['value'] = True

	monkeypatch.setattr(calculator, '_plot', mock_plot)

	calculator.correlation(sc_context, output_path=None)
	assert called['value'] is False


def test_correlation_example_demo():
	# Just test it runs without exception
	CorrelationCalculator.example_demo(show_plot=False)

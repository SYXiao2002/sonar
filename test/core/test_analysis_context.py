import dataclasses
import pytest
from sonar.core.analysis_context import AnalysisContext, SubjectChannel
from sonar.core.dataset_loader import DatasetLoader
from sonar.core.region_selector import RegionSelector
from sonar.core.window_selector import WindowSelector


@pytest.fixture
def dummy_dataset():
	return DatasetLoader.generate_simulated_hbo(
		subject_label_l=["Sub01", "Sub02"],
		n_channels=4,
		sr=10,
		duration=10
	)


def test_subject_channel_immutable():
	sc = SubjectChannel(sub_idx=0, ch_idx=1)
	assert sc.sub_idx == 0
	assert sc.ch_idx == 1

	with pytest.raises(dataclasses.FrozenInstanceError):
		sc.ch_idx = 2  # Should not allow modification


def test_analysis_context_fields(dummy_dataset):
	window_selector = WindowSelector(window_size=2, step=1)
	region_selector = RegionSelector(center_sec=5, length_sec=4)

	context = AnalysisContext(
		dataset=dummy_dataset,
		window_config=window_selector,
		region_selector=region_selector
	)

	assert context.dataset is dummy_dataset
	assert context.window_config is window_selector
	assert context.region_selector is region_selector

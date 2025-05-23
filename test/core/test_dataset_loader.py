import pytest
import numpy as np

from sonar.core.dataset_loader import DatasetLoader


def test_dataset_loader_init_and_getitem():
	# Prepare inputs
	labels = ["subj1", "subj2"]
	n_channels = 3
	sr = 10
	duration = 2  # seconds
	n_timepoints = int(sr * duration)
	time = np.linspace(0, duration, n_timepoints)

	# Simulate data: 2 subjects, each with 3 channels, each channel with n_timepoints samples
	data_l = []
	for _ in labels:
		subject_data = [np.random.rand(n_timepoints) for _ in range(n_channels)]
		data_l.append(subject_data)

	# Instantiate DatasetLoader
	ds = DatasetLoader(time, data_l, labels, sr)

	# Check attributes
	assert np.allclose(ds.raw_time, time)
	assert ds.label_l == labels
	assert ds.sr == sr

	# __getitem__ for "time"
	assert np.allclose(ds["time"], time)

	# __getitem__ by index
	assert np.allclose(ds[0][0], data_l[0][0])
	assert np.allclose(ds[1][2], data_l[1][2])

	# __getitem__ by label
	assert np.allclose(ds["subj1"][1], data_l[0][1])
	assert np.allclose(ds["subj2"][0], data_l[1][0])

	# __repr__ contains labels preview
	rep = repr(ds)
	for label in labels:
		assert label in rep

def test_dataset_loader_invalid_inputs():
	time = np.linspace(0, 1, 10)

	# data_l not list of lists
	with pytest.raises(AssertionError):
		DatasetLoader(time, [np.random.rand(10)], ["subj1"], 10)

	# length mismatch between data and labels
	with pytest.raises(AssertionError):
		DatasetLoader(time, [[np.random.rand(10)]], ["subj1", "subj2"], 10)

	# time length mismatch in inner channel
	with pytest.raises(AssertionError):
		DatasetLoader(time, [[[1,2,3], [1,2]]], ["subj1"], 10)

def test_generate_simulated_hbo():
	labels = ["s1", "s2"]
	n_channels = 4
	sr = 5
	duration = 1.0
	ds = DatasetLoader.generate_simulated_hbo(labels, n_channels, sr, duration, seed=42)

	# Check types and sizes
	assert len(ds.data_l) == len(labels)
	assert len(ds.data_l[0]) == n_channels
	assert len(ds.raw_time) == int(sr * duration)
	assert ds.label_l == labels
	assert ds.sr == sr

	# Check __getitem__ by label
	for label in labels:
		assert label in ds.label_l
		assert ds[label] is not None

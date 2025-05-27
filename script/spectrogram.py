import numpy as np
import dataset as ds
from sonar.core.region_selector import RegionSelector
from sonar.utils.spectrogram_calculator import SpectrogramPlotter


def main():
	dataset, annotations = ds.get_test_dataset(load_cache=True)

	region_selector_l = [
		RegionSelector(start_sec=2250, length_sec=800),
		# RegionSelector(center_sec=2300, length_sec=80),
		# RegionSelector(center_sec=2300, length_sec=40),
		# RegionSelector(center_sec=2300, length_sec=20),
		# RegionSelector(center_sec=2300, length_sec=10),
		# RegionSelector(center_sec=2300, length_sec=5),
	]

	data = dataset[0][0]
	fs=11
	sp = SpectrogramPlotter(fs=fs, window='hann', nperseg=128, noverlap=100, nfft=256,)
	sp.plot(np.array(data))


if __name__ == "__main__":
	main()
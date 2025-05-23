from tqdm import tqdm 

from script.dataset import get_dataset, get_song_list, get_wh_dataset
from sonar.analysis.trend_topomap import TrendTopomap
from sonar.core.region_selector import RegionSelector
from sonar.core.window_selector import WindowSelector


def main():
	# annotations = get_song_list('res/20250510wh_test/marker.csv')
	dataset = get_wh_dataset(debug=False, load_cache=True)
	dataset = get_dataset(debug=False, load_cache=True)
	annotations = get_song_list('res/sv-marker/song_list.csv')

	region_selector_l = [
		RegionSelector(start_sec=2250, length_sec=800),
		# RegionSelector(center_sec=2300, length_sec=80),
		# RegionSelector(center_sec=2300, length_sec=40),
		# RegionSelector(center_sec=2300, length_sec=20),
		# RegionSelector(center_sec=2300, length_sec=10),
		# RegionSelector(center_sec=2300, length_sec=5),
	]


	min_duration_l = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8]

	for min_duration in tqdm(min_duration_l, desc="Computing trend maps"):

		trend_topomap = TrendTopomap(dataset, window_selector=WindowSelector(window_size=1, step=0.1), mode='increasing', min_duration=min_duration, annotations=annotations, region_selector=None, debug=False)
		trend_topomap.save_total_curve_to_csv(output_dir='out')

		for region_selector in tqdm(region_selector_l, desc="Plotting trend maps"):
			trend_topomap.set_region_selector(region_selector)
			trend_topomap.plot_trends(output_dir='out')

if __name__ == "__main__":
	main()
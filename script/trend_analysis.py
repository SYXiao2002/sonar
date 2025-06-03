import os
from tqdm import tqdm 

from sonar.analysis.trend_topomap import TrendTopomap
from sonar.core.dataset_loader import get_dataset
from sonar.core.region_selector import RegionSelector
from sonar.core.window_selector import WindowSelector


def main():
	dataset, annotations = get_dataset(ds_dir='res/trainingcamp-nirspark', load_cache=False)

	region_selector_l = [
		# RegionSelector(center_sec=2740, length_sec=30),
		RegionSelector(start_sec=2250, end_sec=4950),
	]


	min_duration_l = [
		# 0.1, 
		# 0.2,
		# 0.4, 
		# 0.8, 
		1.6, 
		# 3.2, 
		# 6.4, 
		# 12.8
	]

	window_selector = WindowSelector(window_size=1, step=0.1)

	for min_duration in tqdm(min_duration_l, desc="Computing trend maps"):
		csv_dir = f'out/nirspark_min_duation_{min_duration:.1f}s'

		trend_topomap = TrendTopomap(dataset, intensity_window_selector=window_selector, mode='increasing', min_duration=min_duration, annotations=annotations, region_selector=None, debug=False)
		trend_topomap._save_intensity_to_csv(output_dir=csv_dir)

		for region_selector in tqdm(region_selector_l, desc="Plotting trend maps"):
			fig_dir = os.path.join(csv_dir, f'region_{int(region_selector.start_sec)}s_{int(region_selector.end_sec)}s')

			trend_topomap.set_region_selector(region_selector)
			trend_topomap.plot_trends(output_dir=fig_dir)
			trend_topomap._save_trends_to_csv(save_path='out/test/trends_raw.csv')

def run(ds_dir='test', load_cache='False'):
	dataset, annotations = get_dataset(ds_dir=os.path.join('res', ds_dir), load_cache=load_cache)

	intensity_window_selector = WindowSelector(window_size=1, step=0.1)

	# region_selector = RegionSelector(start_sec=2230, end_sec=570)
	region_selector = None

	trend_topomap = TrendTopomap(output_dir=os.path.join('out', ds_dir), 
							  dataset=dataset, intensity_window_selector=intensity_window_selector, 
							  mode='increasing', min_duration=1, 
							  annotations=annotations, region_selector=region_selector, debug=False,
							  high_intensity_thr=30,
							  heartrate_dir='res/heartrate/trainingCamp-mne')

	trend_topomap.plot_trends()
	trend_topomap.plot_high_intensity()


if __name__ == "__main__":
	# run(ds_dir='wh_test', load_cache=False)
	run(ds_dir='trainingcamp-pure', load_cache=False)
	# run(ds_dir='test', load_cache=True)
	
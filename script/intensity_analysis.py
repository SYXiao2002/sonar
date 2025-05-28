import os
from tqdm import tqdm 

from sonar.analysis.intensity_analysis import Peak
from sonar.analysis.trend_topomap import TrendTopomap
from sonar.core import region_selector
from sonar.core.dataset_loader import get_dataset
from sonar.core.region_selector import RegionSelector
from sonar.core.window_selector import WindowSelector


def run(region_selector_l, min_duration=1.6):
	csv_dir = f'out/min_duation_{min_duration:.1f}s'
	dataset, annotations = get_dataset(ds_dir='res/trainingcamp-pure-HC1', load_cache=True)
	window_selector = WindowSelector(window_size=1, step=0.1)

	trend_topomap = TrendTopomap(dataset, window_selector=window_selector, mode='increasing', min_duration=min_duration, annotations=annotations, region_selector=None, debug=False)
	trend_topomap.save_intensity_to_csv(output_dir=csv_dir)

	for region_selector in tqdm(region_selector_l, desc="Plotting trend maps"):
		# fig_dir = os.path.join(csv_dir, f'region_{int(region_selector.start_sec)}s_{int(region_selector.end_sec)}s')
		fig_dir = os.path.join(csv_dir, 'all')

		trend_topomap.set_region_selector(region_selector)
		trend_topomap.plot_trends(output_dir=fig_dir)



if __name__ == "__main__":
	intensity_path = 'out/min_duation_1.6s/intensity_peaks.csv'
	peak_l = Peak.read_peaks_from_csv(intensity_path)

	region_selector_l = []
	for peak in peak_l:
		region_selector_l.append(RegionSelector(start_sec=peak.start_sec-30, end_sec=peak.end_sec+30))

	run(region_selector_l, min_duration=1.6)
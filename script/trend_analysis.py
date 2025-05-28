import csv
import os
from tqdm import tqdm 

from sonar.analysis.intensity_analysis import IntensityAnalyzer
from sonar.analysis.trend_topomap import TrendTopomap
from sonar.core.dataset_loader import get_dataset
from sonar.core.region_selector import RegionSelector
from sonar.core.window_selector import WindowSelector


def main():
	dataset, annotations = get_dataset(ds_dir='res/trainingcamp-nirspark', load_cache=False)

	region_selector_l = [
		# RegionSelector(center_sec=2740, length_sec=30),
		RegionSelector(start_sec=2250, length_sec=850),
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

		trend_topomap = TrendTopomap(dataset, window_selector=window_selector, mode='increasing', min_duration=min_duration, annotations=annotations, region_selector=None, debug=False)
		trend_topomap.save_intensity_to_csv(output_dir=csv_dir)

		for region_selector in tqdm(region_selector_l, desc="Plotting trend maps"):
			fig_dir = os.path.join(csv_dir, f'region_{int(region_selector.start_sec)}s_{int(region_selector.end_sec)}s')

			trend_topomap.set_region_selector(region_selector)
			trend_topomap.plot_trends(output_dir=fig_dir)

def intensity_peaks():	
	csv_path = 'out/min_duation_1.6s/intensity_increasing_HC1_1.6s.csv'

	times = []
	values = []
	with open(csv_path, newline='', encoding='utf-8-sig') as f:
		reader = csv.DictReader(f)
		for row in reader:
			times.append(float(row['time']))
			values.append(float(row['value']))

	analyzer = IntensityAnalyzer(times, values, smooth_size=30, threshold=30)
	analyzer.save('out/intensity_peaks.csv')

if __name__ == "__main__":
	main()
	# intensity_peaks()
	
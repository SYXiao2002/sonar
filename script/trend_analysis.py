import csv
import os
from tqdm import tqdm 

from script import mts_analysis
from sonar.analysis.intensity_analysis import IntensityAnalyzer
from sonar.analysis.trend_topomap import TrendTopomap, save_binary_ts_by_subject
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

		trend_topomap = TrendTopomap(dataset, window_selector=window_selector, mode='increasing', min_duration=min_duration, annotations=annotations, region_selector=None, debug=False)
		trend_topomap.save_intensity_to_csv(output_dir=csv_dir)

		for region_selector in tqdm(region_selector_l, desc="Plotting trend maps"):
			fig_dir = os.path.join(csv_dir, f'region_{int(region_selector.start_sec)}s_{int(region_selector.end_sec)}s')

			trend_topomap.set_region_selector(region_selector)
			trend_topomap.plot_trends(output_dir=fig_dir)
			trend_topomap.save_trends_to_csv(save_path='out/test/trends_raw.csv')

def single(out_dir):
	dataset, annotations = get_dataset(ds_dir=out_dir, load_cache=False)
	folder_name = os.path.basename(out_dir)
	
	window_selector = WindowSelector(window_size=1, step=0.1)
	min_duration = 1.6
	region_selector = RegionSelector(start_sec=2250, end_sec=4950)

	out_dir = os.path.join('out', f'{folder_name}_min_duation_{min_duration:.1f}s')
	os.makedirs(out_dir, exist_ok=True)

	trends_fig_dir = os.path.join(out_dir, f'trends_fig')
	intensity_raw_dir = os.path.join(out_dir, f'intensity_raw')
	binery_trends_dir = os.path.join(out_dir, f'binery_trends_raw')
	trends_raw_csv = os.path.join(out_dir, f'trends_raw.csv')
	peaks_raw_dir = os.path.join(out_dir, f'peaks_raw')
	peaks_fig_dir = os.path.join(out_dir, f'peaks_fig')
	ch_count_heatmap_dir = os.path.join(out_dir, f'ch_count_heatmap')

	trend_topomap = TrendTopomap(dataset, window_selector=window_selector, mode='increasing', min_duration=min_duration, annotations=annotations, region_selector=None, debug=False)
	trend_topomap.save_intensity_to_csv(output_dir=intensity_raw_dir)

	trend_topomap.set_region_selector(region_selector)
	trend_topomap.plot_trends(output_dir=trends_fig_dir)
	trend_topomap.save_trends_to_csv(save_path=trends_raw_csv)

	save_binary_ts_by_subject(trends_raw_csv, output_dir=binery_trends_dir, sample_rate=10)

	# save intensity peaks
	IntensityAnalyzer.extract_peaks(
		csv_path=os.path.join(intensity_raw_dir, f'intensity_increasing_ALL_{min_duration}s.csv'),
		peaks_raw_dir=peaks_raw_dir,
		peaks_fig_dir=peaks_fig_dir
	)

	# generate heatmap
	for sub_label in dataset.label_l:
		peaks_csv_path = os.path.join(peaks_raw_dir, f'{sub_label}.csv')
		binary_csv_path = os.path.join(binery_trends_dir, f'{sub_label}.csv')
		dict = mts_analysis.compute_channel_event_participation(peaks_csv_path, binary_csv_path)
		mts_analysis.plot_ch_count_heatmap(dict, sub_label, ch_count_heatmap_dir)
		


if __name__ == "__main__":
	# main()
	single('res/trainingcamp-nirspark')
	
import os
from tqdm import tqdm 

from sonar.analysis.high_density_analysis import HighDensityAnalyzer
from sonar.analysis.trend_topomap import TrendTopomap
from sonar.analysis.waveform_topomap import WaveformTopomap
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

		trend_topomap = TrendTopomap(dataset, density_window_selector=window_selector, mode='increasing', min_duration=min_duration, annotations=annotations, region_selector=None, debug=False)
		trend_topomap._save_density_to_csv(output_dir=csv_dir)

		for region_selector in tqdm(region_selector_l, desc="Plotting trend maps"):
			fig_dir = os.path.join(csv_dir, f'region_{int(region_selector.start_sec)}s_{int(region_selector.end_sec)}s')

			trend_topomap.set_region_selector(region_selector)
			trend_topomap.plot_trends(output_dir=fig_dir)
			trend_topomap._save_trends_to_csv(save_path='out/test/trends_raw.csv')

def run(ds_dir='test', load_cache='False', heartrate_dir=None, region_selector_l=None, marker_file=None):
	ds, annotations = get_dataset(ds_dir=os.path.join('res', ds_dir), load_cache=load_cache, marker_file=marker_file)

	intensity_window_selector = WindowSelector(window_size=1, step=0.1)


	if region_selector_l is None:
		region_selector_l = [
			None,
			# RegionSelector(start_sec=2230, length_sec=600),
			# RegionSelector(center_sec=2752, length_sec=100), #回马枪

		]

	# waveform_topomap = WaveformTopomap(ds, region_selector_l[0], 'res/test/snirf_metadata.csv', os.path.join('out', ds_dir, 'fig_waveform_topomap'))
	# waveform_topomap.plot(['HC1', 'HC3', 'HC5'])
	# return

	trend_topomap = TrendTopomap(output_dir=os.path.join('out', ds_dir), 
							  dataset=ds, density_window_selector=intensity_window_selector, 
							  mode='increasing', min_duration=1, 
							  annotations=annotations, region_selector=None, debug=False,
							  high_density_thr=30, max_value=None,
							  heartrate_dir=heartrate_dir)

	for r in region_selector_l:
		trend_topomap.set_region_selector(r)
		trend_topomap.plot_trends()
	# trend_topomap.permutation_test()
	# trend_topomap.plot_high_intensity()

if __name__ == "__main__":
	pass
	run(ds_dir='trainingcamp-mne-april', load_cache=False, heartrate_dir='res/trainingcamp-mne-no-filter/spectrogram', 
	 	marker_file='res/trainingcamp-mne-april/marker/mild&intense.csv',
		region_selector_l=[
			RegionSelector(start_sec=2222, length_sec=585), 
			RegionSelector(start_sec=3487, length_sec=377),
			RegionSelector(start_sec=4027, length_sec=363),
			RegionSelector(start_sec=4763, length_sec=221),
	])
	run(ds_dir='trainingcamp-homer3', load_cache=False, heartrate_dir='res/trainingcamp-mne-no-filter/spectrogram', 
	 	marker_file='res/trainingcamp-mne-april/marker/mild&intense.csv',
		region_selector_l=[
			RegionSelector(start_sec=2222, length_sec=585), 
			RegionSelector(start_sec=3487, length_sec=377),
			RegionSelector(start_sec=4027, length_sec=363),
			RegionSelector(start_sec=4763, length_sec=221),
	])
	# run(ds_dir='test', load_cache=True)
	# run(ds_dir='wh_test', load_cache=False)
	# run(ds_dir='trainingcamp-pure', load_cache=True, heartrate_dir='res/trainingcamp-mne-no-filter/spectrogram')
	# run(ds_dir='yuanqu-mne', load_cache=True, heartrate_dir='res/yuanqu-mne-no-filter/spectrogram', region_selector_l=[
		# None
	# ])
	# run(ds_dir='trainingcamp-mne-april', load_cache=True, heartrate_dir='res/trainingcamp-mne-no-filter/spectrogram', region_selector_l=[
		# RegionSelector(start_sec=2230, length_sec=600), RegionSelector(center_sec=2752, length_sec=100)
	# ])
	# run(ds_dir='trainingcamp-mne-may', load_cache=True, heartrate_dir='res/trainingcamp-mne-no-filter/spectrogram')
	# run(ds_dir='trainingcamp-mne-june', load_cache=True, heartrate_dir='res/trainingcamp-mne-no-filter/spectrogram')
	# run(ds_dir='trainingcamp-nirspark', load_cache=True, heartrate_dir='res/trainingcamp-mne-no-filter/spectrogram', region_selector_l=[
		# RegionSelector(start_sec=2230, length_sec=600)
	# ])
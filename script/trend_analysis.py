import os
from tqdm import tqdm 

from sonar.analysis.high_density_analysis import HighDensityAnalyzer
from sonar.analysis.trend_topomap import TrendTopomap
from sonar.analysis.waveform_topomap import WaveformTopomap
from sonar.core.dataset_loader import get_dataset
from sonar.core.region_selector import RegionSelector
from sonar.core.window_selector import WindowSelector

def run(ds_dir='test', load_cache='False', heartrate_dir=None, region_selector_l=None, marker_file=None, high_density_thr=30):
	ds, annotations = get_dataset(ds_dir=os.path.join('res', ds_dir), load_cache=load_cache, marker_file=marker_file)

	intensity_window_selector = WindowSelector(window_size=1, step=0.2)


	if region_selector_l is None:
		region_selector_l = [
			None,
			# RegionSelector(start_sec=2230, length_sec=600),
			# RegionSelector(center_sec=2752, length_sec=100), #回马枪

		]

	trend_topomap = TrendTopomap(output_dir=os.path.join('out', ds_dir), 
							  dataset=ds, density_window_selector=intensity_window_selector, 
							  mode='increasing', min_duration=1, 
							  annotations=annotations, region_selector=None, debug=False,
							  high_density_thr=high_density_thr, max_value=None,
							  heartrate_dir=heartrate_dir)

	for r in region_selector_l:
		trend_topomap.set_region_selector(r)
		trend_topomap.plot_trends(metadata_path=os.path.join('res', ds_dir, 'snirf', 'snirf_metadata.csv'))
	# trend_topomap.permutation_test()
	# trend_topomap.plot_high_density()

if __name__ == "__main__":
	# run(ds_dir='tapping-luke-april', load_cache=False, high_density_thr=20,
	# 	region_selector_l=[
	# 		RegionSelector(start_sec=400*i, length_sec=400) for i in range(6)
	# ])
	# run(ds_dir='trainingcamp-mne-april', load_cache=False, high_density_thr=30,
	# 	region_selector_l=[
	# 		RegionSelector(start_sec=2222+400*i, length_sec=400) for i in range(6)
	# ])

	# run(ds_dir='trainingcamp-mne-luke', load_cache=False, heartrate_dir='res/trainingcamp-mne-no-filter/spectrogram', 
	#  	marker_file='res/trainingcamp-mne-april/marker/mild&intense.csv',
	# 	region_selector_l=[
	# 		RegionSelector(start_sec=2264, length_sec=210), 
	# ])

	# run(ds_dir='trainingcamp-mne-april', load_cache=True, heartrate_dir='res/trainingcamp-mne-no-filter/spectrogram', 
	#  	marker_file='res/trainingcamp-mne-april/marker/mild&intense.csv',
	# 	region_selector_l=[
	# 		RegionSelector(start_sec=2264, length_sec=210), 
	# ])

	# run(ds_dir='trainingcamp-homer3', load_cache=False, heartrate_dir='res/trainingcamp-mne-no-filter/spectrogram', 
	#  	marker_file='res/trainingcamp-mne-april/marker/mild&intense.csv',
	# 	region_selector_l=[
	# 		RegionSelector(start_sec=2264, length_sec=210), 
	# ])
	# run(ds_dir='trainingcamp-mne-nirspark-homer3', load_cache=True)
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
	pass
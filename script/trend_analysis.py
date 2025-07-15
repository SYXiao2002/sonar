import os

from sonar.analysis.trend_topomap import TrendTopomap
from sonar.core.dataset_loader import get_dataset
from sonar.core.region_selector import RegionSelector
from sonar.core.window_selector import WindowSelector

def main(ds_dir, load_cache, heartrate_dir=None, region_selector_l=None, marker_file=None, high_density_thr=30):
	ds, annotations = get_dataset(ds_dir=os.path.join('res', ds_dir), load_cache=load_cache, marker_file=marker_file)
 
	intensity_window_selector = WindowSelector(window_size=1, step=0.3)


	if region_selector_l is None:
		region_selector_l = [
			RegionSelector(center_sec=2752, length_sec=100), #回马枪
		]

	trend_topomap = TrendTopomap(output_dir=os.path.join('out', ds_dir), 
							  dataset=ds, density_window_selector=intensity_window_selector, 
							  mode='increasing', min_duration=1, 
							  annotations=annotations, region_selector=None, debug=False,
							  high_density_thr=high_density_thr, max_value=None,
							  heartrate_dir=heartrate_dir,
							  metadata_path=os.path.join('res', ds_dir, 'snirf', 'snirf_metadata.csv'))

	for r in region_selector_l:
		trend_topomap.set_region_selector(r)
		trend_topomap.plot_trends()

if __name__ == "__main__":
	main(ds_dir='trainingcamp-mne-april', 
	 	load_cache=True,
		heartrate_dir='res/trainingcamp-mne-no-filter/spectrogram'
	)
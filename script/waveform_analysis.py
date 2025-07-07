import os

from sonar.analysis.waveform_topomap import WaveformTopomap
from sonar.core.dataset_loader import get_dataset
from sonar.core.region_selector import RegionSelector


def main(ds_dir, use_raw, sub_label_l, region_selector, suptitle):
	ds, _ = get_dataset(ds_dir=os.path.join('res', ds_dir), load_cache=True, use_raw=use_raw)

	waveform_topomap = WaveformTopomap(ds, region_selector, metadata_path=os.path.join('res', ds_dir, 'snirf', 'snirf_metadata.csv'), output_dir=os.path.join('out', ds_dir))
	waveform_topomap.plot(sub_label_l, suptitle=suptitle)

if __name__ == '__main__':
	main(ds_dir='trainingcamp-mne-april', 
	  use_raw=True, 
	  region_selector=RegionSelector(center_sec=2755, length_sec=20), 
	  sub_label_l=['HC1', 'HC3', 'HC5'],
	  suptitle='MNE-April')
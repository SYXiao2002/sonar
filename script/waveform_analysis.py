import os
from typing import Sequence
from sonar.analysis.waveform_topomap import WaveformTopomap
from sonar.core import region_selector
from sonar.core.dataset_loader import get_dataset
from sonar.core.region_selector import RegionSelector
from sonar.preprocess.sv_marker import Annotation, read_annotations


def run(ds_dir = 'trainingcamp-mne-april'):
	ds, _ = get_dataset(ds_dir=os.path.join('res', ds_dir), load_cache=True)
	metadata_path = 'res/test/snirf_metadata.csv'

	annotation: Sequence[Annotation] = read_annotations('res/trainingcamp-mne-april/marker/test.csv')


	# performer
	# performer_sub_label_l = ['HC7', 'HC9']
	# for a in annotation:
		# region_selector = RegionSelector(start_sec=a.start, length_sec=a.duration)
		# waveform_topomap = WaveformTopomap(ds, region_selector, metadata_path, os.path.join('out', 'segments-view', 'performer', a.label))
		# waveform_topomap.plot(performer_sub_label_l, suptitle=f'Performer Waveform Topomap: {a.label} ( length = {a.duration:.0f} sec )\n Dataset: filter(0.01hz, 0.09hz), Nirspark')

	# audience
	audience_sub_label_l = ['HC1', 'HC3', 'HC5']
	for a in annotation:
		region_selector = RegionSelector(start_sec=a.start, length_sec=a.duration)
		for sub_label in audience_sub_label_l:
			waveform_topomap = WaveformTopomap(ds, region_selector, metadata_path, os.path.join('out', ds_dir, 'fig_waveform', sub_label , a.label))
			waveform_topomap.plot([sub_label], suptitle=f'Waveform Topomap: {ds_dir}-{sub_label} ( length = {a.duration:.0f} sec )')


if __name__ == '__main__':
	run(ds_dir='trainingcamp-nirspark')
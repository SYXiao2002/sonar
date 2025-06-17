import os
from typing import Sequence

from matplotlib.pyplot import ylim
from sonar.analysis.waveform_topomap import WaveformTopomap
from sonar.core import region_selector
from sonar.core.dataset_loader import get_dataset
from sonar.core.region_selector import RegionSelector
from sonar.preprocess.sv_marker import Annotation, read_annotations


def run(ds_dir, use_raw, ylim):
	ds, _ = get_dataset(ds_dir=os.path.join('res', ds_dir), load_cache=True, use_raw=use_raw)
	metadata_path = 'res/test/snirf_metadata.csv'

	roi_annotation: Sequence[Annotation] = read_annotations('res/trainingcamp-mne-nirspark-homer3/marker/test.csv')
	# topomap_annotation: Sequence[Annotation] = read_annotations('res/trainingcamp-mne-nirspark-homer3/marker/mild&intense.csv')
	topomap_annotation: Sequence[Annotation] = read_annotations('res/trainingcamp-homer3/marker/marker.csv')


	# performer
	# performer_sub_label_l = ['HC7', 'HC9']
	# for a in annotation:
		# region_selector = RegionSelector(start_sec=a.start, length_sec=a.duration)
		# waveform_topomap = WaveformTopomap(ds, region_selector, metadata_path, os.path.join('out', 'segments-view', 'performer', a.label))
		# waveform_topomap.plot(performer_sub_label_l, suptitle=f'Performer Waveform Topomap: {a.label} ( length = {a.duration:.0f} sec )\n Dataset: filter(0.01hz, 0.09hz), Nirspark')

	# audience
	# audience_sub_label_l = ['HC1', 'HC3', 'HC5']
	# for a in annotation:
		# region_selector = RegionSelector(start_sec=a.start, length_sec=a.duration)
		# for sub_label in audience_sub_label_l:
			# waveform_topomap = WaveformTopomap(ds, region_selector, metadata_path, os.path.join('out', ds_dir, 'fig_waveform', sub_label , a.label))
			# waveform_topomap.plot([sub_label], suptitle=f'Waveform Topomap: {ds_dir}-{sub_label} ( length = {a.duration:.0f} sec )')

	# mixed
	sub_label_l = ['homer3', 'mne', 'nirspark']
	# sub_label_l = ['homer3']
	for a in roi_annotation:
		region_selector = RegionSelector(start_sec=a.start, length_sec=a.duration)
		waveform_topomap = WaveformTopomap(ds, region_selector, metadata_path, os.path.join('out', ds_dir, 'fig_waveform', a.label))
		waveform_topomap.plot(sub_label_l, suptitle=f'Waveform Topomap: HC1 ( length = {a.duration:.0f} sec )', annotation=topomap_annotation, ylim=ylim)

if __name__ == '__main__':
	# run(ds_dir='trainingcamp-nirspark', use_raw=False)
	# run(ds_dir='trainingcamp-mne-nirspark-homer3', use_raw=True, ylim=(-2*1e-6, 2*1e-6))
	run(ds_dir='trainingcamp-mne-nirspark-homer3', use_raw=False, ylim=(-2, 2))
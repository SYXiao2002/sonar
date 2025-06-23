import os
from typing import Sequence

from matplotlib import use

from sonar.analysis.waveform_topomap import WaveformTopomap
from sonar.core import region_selector
from sonar.core.dataset_loader import get_dataset
from sonar.core.region_selector import RegionSelector
from sonar.preprocess.sv_marker import Annotation, read_annotations


def run(ds_dir, use_raw):
	ds, _ = get_dataset(ds_dir=os.path.join('res', ds_dir), load_cache=True, use_raw=use_raw)
	metadata_path = os.path.join('res', ds_dir, 'snirf', 'snirf_metadata.csv')

	topomap_annotation: Sequence[Annotation] = read_annotations(os.path.join('res', ds_dir, 'marker', 'marker.csv'))
	# topomap_annotation: Sequence[Annotation] = read_annotations('res/trainingcamp-homer3/marker/marker.csv')

	sub_label = 'sub01'

	# for i in range(6):
	# 	region_selector = RegionSelector(start_sec=400*i, length_sec=400)
	# 	waveform_topomap = WaveformTopomap(ds, region_selector, metadata_path, os.path.join('out', ds_dir, 'fig_waveform', '400s'))
	# 	waveform_topomap.plot([sub_label], use_raw=use_raw,  suptitle=f'Waveform Topomap: ( length = {400:.0f} sec )\n Dataset: filter(0.05hz, 0.7hz), MNE', annotation=topomap_annotation, expand=5)

	# roi_annotation: Sequence[Annotation] = read_annotations('out/tapping-luke-april/raw_high_density/sub01.csv')
	# for roi in roi_annotation:
	# 	region_selector = RegionSelector(center_sec=roi.start+roi.duration/2, length_sec=5)
	# 	waveform_topomap = WaveformTopomap(ds, region_selector, metadata_path, os.path.join('out', ds_dir, 'fig_waveform', sub_label , 'high_density'))
	# 	waveform_topomap.plot([sub_label], use_raw=use_raw,  suptitle=f'Waveform Topomap: {roi.label} ( length = {roi.duration:.0f} sec )\n Dataset: filter(0.05hz, 0.7hz), MNE', annotation=topomap_annotation, expand=5)

	roi_annotation: Sequence[Annotation] = read_annotations(os.path.join('res', ds_dir, 'marker', 'marker.csv'))
	for roi in roi_annotation:
		region_selector = RegionSelector(start_sec=roi.start, length_sec=5)
		waveform_topomap = WaveformTopomap(ds, region_selector, metadata_path, os.path.join('out', ds_dir, 'fig_waveform', sub_label , roi.label))
		waveform_topomap.plot([sub_label], use_raw=use_raw,  suptitle=f'Waveform Topomap: {roi.label} ( length = {roi.duration:.0f} sec )\n Dataset: filter(0.02hz, 0.09hz), MNE', annotation=topomap_annotation, expand=5)

	
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
	# sub_label_l = ['homer3', 'mne', 'nirspark']

	# region_selector = RegionSelector(start_sec=2264, length_sec=210)
	# raw_marker = 'raw' if use_raw  else 'z'
	# waveform_topomap = WaveformTopomap(ds, region_selector, metadata_path, os.path.join('out', ds_dir, 'fig_waveform', f'{raw_marker}'))
	# waveform_topomap.plot(sub_label_l, use_raw=use_raw, suptitle=f'Waveform Topomap: HC1 {raw_marker}( length = {region_selector.length_sec:.0f} sec )')


	# sub_label_l = ['homer3', 'mne', 'nirspark']
	# for sub_label in sub_label_l:
	# 	waveform_topomap = WaveformTopomap(ds, region_selector, metadata_path, os.path.join('out', ds_dir, 'fig_waveform', f'{sub_label}'))
	# 	waveform_topomap.plot([sub_label], use_raw=use_raw, suptitle=f'Waveform Topomap: HC1 {raw_marker}( length = {region_selector.length_sec:.0f} sec )')

	return 

	# single
	sub_label = 'HC1'
	topomap_annotation: Sequence[Annotation] = read_annotations(os.path.join('out', ds_dir, 'raw_high_density', f'{sub_label}.csv'))
	for t in topomap_annotation:
		t = t._replace(label='MTS')

	region_selector = RegionSelector(start_sec=2264, length_sec=210)
	raw_marker = 'raw' if use_raw  else 'z'
	waveform_topomap = WaveformTopomap(ds, region_selector, metadata_path, os.path.join('out', ds_dir, 'fig_waveform', f'{raw_marker}'))
	waveform_topomap.plot([sub_label], use_raw=use_raw, suptitle=f'Waveform Topomap: HC1 {raw_marker}( length = {region_selector.length_sec:.0f} sec )', annotation=topomap_annotation)

def main(ds_dir, use_raw):
	ds, _ = get_dataset(ds_dir=os.path.join('res', ds_dir), load_cache=False)
	region_selector = RegionSelector(start_sec=0, length_sec=500)
	waveform_topomap = WaveformTopomap(ds, region_selector, metadata_path=os.path.join('res', ds_dir, 'snirf', 'snirf_metadata.csv'), output_dir=os.path.join('out', ds_dir))
	waveform_topomap.plot(['Sub5'], use_raw=use_raw, suptitle=f'Waveform Topomap: Sub5, Cortivision ( length = {region_selector.length_sec:.0f} sec )')

if __name__ == '__main__':
	# main(ds_dir='trainingcamp-mne-no-filter', use_raw=True)
	# run(ds_dir='trainingcamp-nirspark', use_raw=False)
	# run(ds_dir='trainingcamp-homer3', use_raw=False)
	# run(ds_dir='tapping-luke-april', use_raw=True)
	# run(ds_dir='trainingcamp-mne-april', use_raw=True)
	main(ds_dir='yuanqu-mne-cortivision', use_raw=True)
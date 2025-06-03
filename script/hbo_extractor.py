
from itertools import cycle
import os
from typing import Sequence

from matplotlib import pyplot as plt
from matplotlib import lines
from matplotlib.lines import lineStyles

from sonar.core.analysis_context import SubjectChannel
from sonar.core.dataset_loader import extract_hbo, get_dataset
from sonar.core.region_selector import RegionSelector
from sonar.preprocess.sv_marker import Annotation, read_annotations


def run(ds_dir='test', load_cache='False'):
	dataset, annotations = get_dataset(ds_dir=os.path.join('res', ds_dir), load_cache=False)

	extract_hbo(dataset, output_dir=os.path.join('out', ds_dir))

def plot_solo_seg(ds_dir):
	ds, _ = get_dataset(ds_dir=os.path.join('res', ds_dir), load_cache=True)
	seg_annotation: Sequence[Annotation] = read_annotations('res/trainingcamp-svmarker/solo.csv')
	out_dir = os.path.join('out', ds_dir, 'fig_waveform')
	os.makedirs(out_dir, exist_ok=True)

	SubjectChannel_l = [
		{'sub_label': 'HC9', 'ch_idx': 48, 'desc': 'cello, pre-motor(ch48)'},	#cello, left, pre-motor 81% (Talairach daemon)
		{'sub_label': 'HC9', 'ch_idx': 35, 'desc': 'celo, auditory(ch35)'},	#cello, left, pre-motor 81% (Talairach daemon)
		{'sub_label': 'HC7', 'ch_idx': 48, 'desc': 'piano, pre-motor(ch48)'},	#cello, left, pre-motor 81% (Talairach daemon)
		{'sub_label': 'HC7', 'ch_idx': 35, 'desc': 'piano, auditory(ch35)'},	#cello, left, pre-motor 81% (Talairach daemon)
	]

	region_selector_l = [
		(RegionSelector(start_sec=seg.start, length_sec=seg.duration), seg.label) for seg in seg_annotation
	]



	# Fallback color cycle (for subjects)
	color_cycler = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
	subject_colors = {}

	# Fallback linestyle cycle (for channels)
	linestyle_cycler = cycle(['-', '--', '-.', ':', (0, (3, 1, 1, 1))])
	channel_linestyles = {}

	# Assign colors to subjects
	for sc in SubjectChannel_l:
		sub_label = sc['sub_label']
		if sub_label not in subject_colors:
			subject_colors[sub_label] = next(color_cycler)

	for r, seg_label in region_selector_l:
		for sc in SubjectChannel_l:
			sub_label = sc['sub_label']
			ch_idx = sc['ch_idx'] - 1
			label = sc['desc']

			# Assign color to subject if not already done
			if sub_label not in subject_colors:
				subject_colors[sub_label] = next(color_cycler)
			color = subject_colors[sub_label]

			# Assign linestyle to channel index if not already done
			if ch_idx not in channel_linestyles:
				channel_linestyles[ch_idx] = next(linestyle_cycler)
			linestyle = channel_linestyles[ch_idx]

			# Plot with auto-assigned color and linestyle
			plt.plot(ds['time'], ds[sub_label][ch_idx],
				label=f'{label}',
				color=color,
				linestyle=linestyle)
		plt.axvspan(r.start_sec, r.end_sec, color='orange', alpha=0.3)
		plt.xlim((r.start_sec-10, r.end_sec+10))
		plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
		plt.legend()
		plt.title(f'{seg_label}')
		plt.xlabel('Time (s)')
		plt.ylim(-3, 3)
		plt.ylabel('HbO Signal (z-scored)')
		path = os.path.join(out_dir, seg_label)
		os.makedirs(path, exist_ok=True)
		plt.savefig(os.path.join(path, f'{r.start_sec:.0f}-{r.end_sec:.0f}.png'), dpi=600)
		plt.close()

if __name__ == "__main__":
	plot_solo_seg(ds_dir = 'trainingcamp-mne-april')
	# plot_solo_seg(ds_dir = 'trainingcamp-mne-may')
	# plot_solo_seg(ds_dir = 'trainingcamp-nirspark')
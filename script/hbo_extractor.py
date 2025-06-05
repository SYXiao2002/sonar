
from itertools import cycle
import os
from typing import Sequence

from matplotlib import pyplot as plt
from matplotlib import lines
from matplotlib.lines import lineStyles
import numpy as np
import pandas as pd
from tqdm import tqdm

from sonar.core import region_selector
from sonar.core.analysis_context import SubjectChannel
from sonar.core.dataset_loader import DatasetLoader, extract_hbo, get_dataset
from sonar.core.region_selector import RegionSelector
from sonar.preprocess.sv_marker import Annotation, read_annotations


def run(ds_dir='test', load_cache='False'):
	dataset, annotations = get_dataset(ds_dir=os.path.join('res', ds_dir), load_cache=False)

	extract_hbo(dataset, output_dir=os.path.join('out', ds_dir))

def plot_solo_seg(ds_dir, annotation_path, SubjectChannel_l):
	ds, _ = get_dataset(ds_dir=os.path.join('res', ds_dir), load_cache=True)
	seg_annotation: Sequence[Annotation] = read_annotations(annotation_path)
	out_dir = os.path.join('out', ds_dir, 'fig_waveform')
	os.makedirs(out_dir, exist_ok=True)

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


def plot_segments(dataset: DatasetLoader, sub_label_l: Sequence[str], channel_l: Sequence[int], annotation_l: Sequence[Annotation], out_dir):
	df = pd.DataFrame(annotation_l)
	seg_label_dict = {}
	for a in annotation_l:
		seg_label_dict[a.label] = 1

	groups = df.groupby('label')


	for label, group_df in tqdm(groups, desc="Labels"):
		for ch_idx in tqdm(channel_l, desc=f"Channels for {label}", leave=False):
			path = os.path.join(out_dir, label)
			os.makedirs(path, exist_ok=True)
			path = os.path.join(path, f'CH{ch_idx}.png')

			n = len(group_df)
			max_per_row = 4
			n_rows = (n + max_per_row - 1) // max_per_row

			fig, axes = plt.subplots(n_rows, max_per_row, figsize=(16, 3 * n_rows))
			axes = axes.flatten()

			for i, (idx, row) in enumerate(group_df.iterrows()):
				ax = axes[i]
				ax.set_title(f"Subplot {i+1}")
				region_selector = RegionSelector(start_sec=row['start'], length_sec=row['duration'])
				for sub_label in sub_label_l:
					ax.plot(dataset['time'], dataset[sub_label][ch_idx-1], label=sub_label)
				ax.axvspan(region_selector.start_sec, region_selector.end_sec, color='orange', alpha=0.3)
				ax.set_xlim(region_selector.start_sec-10, region_selector.end_sec+10)
				ax.set_xlabel('Time (s)')
				ax.set_ylim(-1, 1)

				xticks = np.arange(region_selector.start_sec-10, region_selector.end_sec + 10 + 1, 5).astype(int)
				ax.set_xticks(xticks)
				ax.set_xticklabels(xticks, rotation=45)
				ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

				if i % max_per_row == 0:
					ax.legend()

			for j in range(n, len(axes)):
				axes[j].axis('off')

			fig.suptitle(f'{label}, CH{ch_idx}')
			plt.tight_layout()
			plt.savefig(path, dpi=600)
			plt.close(fig)  # 关闭图，防止内存泄漏




if __name__ == "__main__":
	ds, _ = get_dataset(ds_dir=os.path.join('res', 'trainingcamp-nirspark'), load_cache=True)
	annotation_l = read_annotations('res/trainingcamp-svmarker/solo.csv')
	channel_l = [35, 48, 40, 43, 38, 20, 18, 16, 10, 11, 29]


	out_dir = os.path.join('out', 'segments-view', 'audience')
	sub_label_l = ['HC1', 'HC3', 'HC5']
	plot_segments(ds, sub_label_l, channel_l,  annotation_l, out_dir=out_dir)


	out_dir = os.path.join('out', 'segments-view', 'performer')
	sub_label_l = ['HC7', 'HC9']
	plot_segments(ds, sub_label_l, channel_l,  annotation_l, out_dir=out_dir)
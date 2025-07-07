"""
File Name: waveform_topomap.py
Author: Yixiao Shen
Date: 2025-06-04
Purpose: waveform topomap
"""

import os

from matplotlib import pyplot as plt, use

from sonar.core.color import get_color_from_label
from sonar.core.dataset_loader import DatasetLoader, get_dataset
from sonar.core.region_selector import RegionSelector
from sonar.preprocess.snirf_metadata import get_metadata_dict, normalize_metadata_pos_dict
from sonar.utils.topomap_plot import plot_anatomical_labels

class WaveformTopomap:
	def __init__(self, dataset, region_selector, metadata_path, output_dir, inset_width=0.07, inset_height=0.10):
		self.dataset: DatasetLoader = dataset
		self.region_selector: RegionSelector = region_selector
		self.metadata_path = metadata_path
		self.output_dir = output_dir
		self.inset_width = inset_width
		self.inset_height = inset_height
		os.makedirs(self.output_dir, exist_ok=True)

	def plot(self, sub_label_l, suptitle=None, expand=0, annotation=None):
		fig = plt.figure(figsize=(12, 8))
		main_ax = fig.add_subplot(111)
		main_ax.axis('off')

		metadata_dict, _ = get_metadata_dict(self.metadata_path)
		metadata_dict = normalize_metadata_pos_dict(metadata_dict, self.inset_width, self.inset_height)

		for i, ch_name in enumerate(self.dataset.ch_l):
			ch_name = ch_name.split()[0]
			ch_idx = metadata_dict[ch_name].idx
			x = metadata_dict[ch_name].pos[0]
			y = metadata_dict[ch_name].pos[1]

			x0 = x - self.inset_width / 2
			y0 = y - self.inset_height / 2
			ax_inset = fig.add_axes([x0, y0, self.inset_width, self.inset_height])
			ax_inset.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')

			for sub_label in sub_label_l:
				color = get_color_from_label(sub_label)
				ax_inset.plot(self.dataset['time'], self.dataset[sub_label][i], label=sub_label, color=color, linewidth=0.7, zorder=3)

			ax_inset.set_ylim(-1, 1)
			ax_inset.grid(True)
			ax_inset.tick_params(axis='both', labelsize=6, direction='in')
			ax_inset.set_title(f'Ch{ch_idx}', fontsize=7, pad=2)

			xticks = self.region_selector.get_integer_ticks(ideal_num_ticks=4)
			ax_inset.set_xticks(xticks)
			ax_inset.set_xticklabels(xticks, rotation=45)  # Rotate 45 degrees

			ax_inset.set_xlim(self.region_selector.get_xlim_range()[0] - expand, self.region_selector.get_xlim_range()[1] + expand)

			if annotation is not None:
				used_labels = set()  # Track labels already used
				for a in annotation:
					color = get_color_from_label(a.label)
					# Only add label if not already used
					if a.label not in used_labels:
						ax_inset.axvspan(
							a.start,
							a.start + a.duration,
							alpha=0.2,
							label=a.label,
							zorder=2,
							color=color
						)
						used_labels.add(a.label)
					else:
						# Don't include label for duplicate entries
						ax_inset.axvspan(
							a.start,
							a.start + a.duration,
							alpha=0.2,
							zorder=2,
							color=color
			)

			# Add legend only to selected channels to avoid visual clutter
			if ch_idx in {47, 48, 46, 44, 41}:
				handles, labels = ax_inset.get_legend_handles_labels()
				ax_inset.legend(
					handles,
					labels,
					fontsize=5,
					loc='center left',
					bbox_to_anchor=(1.01, 0.5),
					frameon=False
				)

			if ch_idx not in {1, 2, 3, 5, 8}:
				ax_inset.set_yticklabels([])
			else:
				ax_inset.set_ylabel('HbO (a.u.)')

		plot_anatomical_labels(plt, template_idx=1)
		if suptitle is not None:
			fig.suptitle(suptitle, fontsize=14)
		plt.tight_layout()
		path = os.path.join(self.output_dir, f'waveform_topomap_{self.region_selector.start_sec:.0f}-{self.region_selector.end_sec:.0f}s.png')
		plt.savefig(path, dpi=600)

	def example():
		sub_label_l = ['test-sub1']
		ds, _ = get_dataset(ds_dir=os.path.join('res', 'test'), load_cache=False, use_raw=False)
		region_selector = RegionSelector(start_sec=2450, length_sec=30)
		metadata_path = 'res/test/snirf/snirf_metadata.csv'
		waveform_topomap = WaveformTopomap(ds, region_selector, metadata_path, os.path.join('out', 'test'))
		waveform_topomap.plot(sub_label_l)

if __name__ == '__main__':
	WaveformTopomap.example()
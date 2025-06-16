"""
File Name: waveform_topomap.py
Author: Yixiao Shen
Date: 2025-06-04
Purpose: waveform topomap
"""

import os

from matplotlib import pyplot as plt
import numpy as np

from sonar.core.dataset_loader import DatasetLoader, get_dataset
from sonar.core.region_selector import RegionSelector
from sonar.utils.topomap_plot import get_meta_data, normalize_positions, plot_anatomical_labels


class WaveformTopomap:
	def __init__(self, dataset, region_selector, metadata_path,  output_dir):
		self.dataset: DatasetLoader = dataset
		self.region_selector: RegionSelector = region_selector
		self.metadata_path = metadata_path
		self.output_dir = output_dir
		os.makedirs(self.output_dir, exist_ok=True)

		self._cal()

	def _cal(self):
		ch_pos_l, _ = get_meta_data(self.metadata_path)
		self.inset_width = 0.078
		self.inset_height = 0.12
		self.ch_pos_l = normalize_positions(ch_pos_l, self.inset_width, self.inset_height, x_range=(0.001, 0.999), y_range=(0.02, 0.95))

		pass

	def plot(self, sub_label_l, suptitle=None, expand=0):
		fig = plt.figure(figsize=(28, 12))
		main_ax = fig.add_subplot(111)
		main_ax.axis('off')

		for ch_idx, (x, y) in enumerate(self.ch_pos_l):
			x0 = x - self.inset_width / 2
			y0 = y - self.inset_height / 2
			ax_inset = fig.add_axes([x0, y0, self.inset_width, self.inset_height])
			ax_inset.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')

			lines = []	# Store Line2D objects for each subject
			for sub_label in sub_label_l:
				line, = ax_inset.plot(self.dataset['time'], self.dataset[sub_label][ch_idx])
				lines.append(line)

			# ax_inset.set_ylim(-1, 1)
			ax_inset.set_ylim(-3, 3)
			ax_inset.grid(True)
			ax_inset.tick_params(axis='both', labelsize=6, direction='in')
			ax_inset.set_title(f'Ch{ch_idx+1}', fontsize=7, pad=2)

			xticks = self.region_selector.get_integer_ticks(ideal_num_ticks=4)
			ax_inset.set_xticks(xticks)
			ax_inset.set_xticklabels(xticks, rotation=45)  # Rotate 45 degrees

			ax_inset.set_xlim(self.region_selector.get_xlim_range()[0] - expand, self.region_selector.get_xlim_range()[1] + expand)
			plt.axvspan(self.region_selector.start_sec, self.region_selector.end_sec, color='orange', alpha=0.2)

			# Add legend only to selected channels to avoid visual clutter
			if ch_idx + 1 in {48, 36, 35, 16, 15}:
				ax_inset.legend(
					lines,
					sub_label_l,
					fontsize=5,
					loc='center left',
					bbox_to_anchor=(1.01, 0.5),
					frameon=False
				)

		plot_anatomical_labels(plt)
		if suptitle is not None:
			fig.suptitle(suptitle, fontsize=14)
		plt.tight_layout()
		path = os.path.join(self.output_dir, f'waveform_topomap_{self.region_selector.start_sec:.0f}-{self.region_selector.end_sec:.0f}s.png')
		plt.savefig(path, dpi=600)

	def example():
		sub_label_l = ['test-sub1']
		ds, _ = get_dataset(ds_dir=os.path.join('res', 'test'), load_cache=False)
		region_selector = RegionSelector(start_sec=2222, length_sec=100)
		metadata_path = 'res/test/snirf_metadata.csv'
		waveform_topomap = WaveformTopomap(ds, region_selector, metadata_path, os.path.join('out', 'test'))
		waveform_topomap.plot(sub_label_l)

if __name__ == '__main__':
	WaveformTopomap.example()
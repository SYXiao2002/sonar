"""
File Name: waveform_topomap.py
Author: Yixiao Shen
Date: 2025-06-04
Purpose: waveform topomap
"""

import hashlib
import os

from matplotlib import pyplot as plt

from sonar.core.dataset_loader import DatasetLoader, get_dataset
from sonar.core.region_selector import RegionSelector
from sonar.utils.topomap_plot import get_meta_data, normalize_positions, plot_anatomical_labels
import seaborn as sns

color_pool = sns.color_palette("tab20")  # 或 "Set2", "Paired", "husl", "dark", "colorblind"
num_colors = len(color_pool)

def get_color_from_label(label):
	# Use md5 for a stable hash
	hash_val = int(hashlib.md5(label.encode()).hexdigest(), 16)
	return color_pool[hash_val % num_colors]

class WaveformTopomap:
	def __init__(self, dataset, region_selector, metadata_path, output_dir):
		self.dataset: DatasetLoader = dataset
		self.region_selector: RegionSelector = region_selector
		self.metadata_path = metadata_path
		self.output_dir = output_dir
		os.makedirs(self.output_dir, exist_ok=True)
		self._cal()

	def _cal(self):
		ch_pos_l, _ = get_meta_data(self.metadata_path)
		self.inset_width = 0.075
		self.inset_height = 0.12
		self.ch_pos_l = normalize_positions(ch_pos_l, self.inset_width, self.inset_height, x_range=(0.001, 0.999), y_range=(0.02, 0.95))

		pass

	def plot(self, sub_label_l, use_raw, suptitle=None, expand=0, annotation=None):
		fig = plt.figure(figsize=(12, 10))
		main_ax = fig.add_subplot(111)
		main_ax.axis('off')

		for ch_idx, (x, y) in enumerate(self.ch_pos_l):
			x0 = x - self.inset_width / 2
			y0 = y - self.inset_height / 2
			ax_inset = fig.add_axes([x0, y0, self.inset_width, self.inset_height])
			ax_inset.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')

			for sub_label in sub_label_l:
				color=get_color_from_label(sub_label)
				ax_inset.plot(self.dataset['time'], self.dataset[sub_label][ch_idx], label=sub_label, color=color, linewidth=0.7, zorder=3)

			ax_inset.set_ylim(-3, 3)
			ax_inset.grid(True)
			ax_inset.tick_params(axis='both', labelsize=6, direction='in')
			ax_inset.set_title(f'Ch{ch_idx+1}', fontsize=7, pad=2)

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
			if ch_idx + 1 in {48, 36, 35, 16, 15}:
				handles, labels = ax_inset.get_legend_handles_labels()
				ax_inset.legend(
					handles,
					labels,
					fontsize=5,
					loc='center left',
					bbox_to_anchor=(1.01, 0.5),
					frameon=False
				)

			if ch_idx + 1 not in {38, 20, 18, 2, 1}:
				ax_inset.set_yticklabels([])
			else:
				if use_raw:
					ax_inset.set_ylabel('HbO (uM)')
				else:
					ax_inset.set_ylabel('HbO (a.u.)')

		plot_anatomical_labels(plt)
		if suptitle is not None:
			fig.suptitle(suptitle, fontsize=14)
		plt.tight_layout()
		path = os.path.join(self.output_dir, f'waveform_topomap_{self.region_selector.start_sec:.0f}-{self.region_selector.end_sec:.0f}s.png')
		plt.savefig(path, dpi=600)
		# plt.show()

	def example():
		sub_label_l = ['test-sub1']
		ds, _ = get_dataset(ds_dir=os.path.join('res', 'test'), load_cache=False)
		region_selector = RegionSelector(start_sec=2222, length_sec=100)
		metadata_path = 'res/test/snirf_metadata.csv'
		waveform_topomap = WaveformTopomap(ds, region_selector, metadata_path, os.path.join('out', 'test'))
		waveform_topomap.plot(sub_label_l)

if __name__ == '__main__':
	WaveformTopomap.example()
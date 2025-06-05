"""
File Name: high_intensity_analysis.py
Author: Yixiao Shen
Date: 2025-06-04
Purpose: mts == high intensity
"""

import os
from collections import Counter
from dataclasses import dataclass
from tqdm import tqdm
from pyparsing import Sequence

import numpy as np
from matplotlib import cm, colors, pyplot as plt
from statsmodels.stats.multitest import multipletests
import pandas as pd

from sonar.analysis.intensity_analysis import HighIntensity
from sonar.utils.topomap_plot import get_meta_data, normalize_positions, plot_anatomical_labels

@dataclass
class ChannelTestResult:
	channel: str
	real_count: int
	real_percent: float
	p_value: float
	p_value_corrected: float
	significant: bool

class HighIntensityAnalyzer:
	def __init__(self, ds_dir):
		high_intensity_dir = os.path.join(ds_dir, 'raw_high_intensity')
		self.high_intensity_csv_l = [os.path.join(high_intensity_dir, f) for f in os.listdir(high_intensity_dir) if f.endswith('.csv')]
		self.sub_label_l = [f.split('.')[0] for f in os.listdir(high_intensity_dir)]
		self.output_dir = ds_dir
		self.results: dict[str, (Sequence[ChannelTestResult], int)] = {}
		
		self._cal()
		self._plot()
		self._save()

	def _cal(self):
		for sub_idx, sub_label in enumerate(self.sub_label_l):
			csv_path = self.high_intensity_csv_l[sub_idx]
			result = peak_permutation_test(csv_path, n_perm=1000, seed=42)
			self.results[sub_label] = result

	def _plot(self):
		output_dir = os.path.join(self.output_dir, 'fig_permutation')
		os.makedirs(output_dir, exist_ok=True)
		for sub_label in self.sub_label_l:
			result = self.results[sub_label][0]
			n=self.results[sub_label][1]
			fig = plt.figure(figsize=(12, 8))
			main_ax = fig.add_subplot(111)
			main_ax.axis('off')

			box_width = 0.07
			box_height = 0.10

			metadata_path = "res/test/snirf_metadata.csv"
			ch_pos_l, ch_name_l = get_meta_data(metadata_path)
			ch_pos_l = normalize_positions(ch_pos_l, box_width, box_height, x_range=(0.02, 0.9), y_range=(0.05, 0.9))

			norm = colors.Normalize(vmin=85, vmax=100)
			cmap = cm.get_cmap('gist_yarg')  # 可换为 'hot' 'viridis' 等

			for ch_idx, (x, y) in enumerate(ch_pos_l):
				x0 = x - box_width / 2
				y0 = y - box_height / 2
				ax_inset = fig.add_axes([x0, y0, box_width, box_height])
				ax_inset.set_xticks([])
				ax_inset.set_yticks([])

				# Set title with significance marker
				val = result[ch_idx].real_percent
				color = cmap(norm(val))
				sig_marker = ''
				if result[ch_idx].p_value_corrected < 0.01:
					sig_marker = '**'
				elif result[ch_idx].p_value_corrected < 0.05:
					sig_marker = '*'

				if result[ch_idx].significant:
					ax_inset.spines['bottom'].set_color('red')
					ax_inset.spines['top'].set_color('red')
					ax_inset.spines['left'].set_color('red')
					ax_inset.spines['right'].set_color('red')
					for spine in ax_inset.spines.values():
						spine.set_linewidth(1.5)

				ax_inset.set_title(f'Ch{ch_idx+1}: {int(val)}% {sig_marker}', fontsize=7, pad=2)
				ax_inset.set_facecolor(color)

			cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
			cbar=plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
			cbar.set_label('Contribution Rate', fontsize=10)

			plot_anatomical_labels(plt, 2)

			plt.suptitle(f'Topomap: Channel Contribution Rate, {sub_label}\nn = {n}(*: p < 0.05   **: p < 0.01)', fontsize=14)
			plt.savefig(os.path.join(output_dir, f"{sub_label}.png"), dpi = 600)

	def _save(self):
		output_dir = os.path.join(self.output_dir, 'raw_permutation')
		wjy_request_dir = os.path.join(self.output_dir, 'wjy_request')
		os.makedirs(output_dir, exist_ok=True)
		os.makedirs(wjy_request_dir, exist_ok=True)
		for sub_label in self.sub_label_l:
			result, _, wjy_request_1k, wjy_request_real = self.results[sub_label]
			csv_path = os.path.join(output_dir, f"{sub_label}.csv")
			df = pd.DataFrame(result)
			df.to_csv(csv_path, index=False)
			df = pd.DataFrame(wjy_request_1k)
			df.to_csv(os.path.join(wjy_request_dir, f"{sub_label}_1k.csv"), index=False)
			df = pd.Series(wjy_request_real).to_frame().T
			df.to_csv(os.path.join(wjy_request_dir, f"{sub_label}_real.csv"), index=False)

def count_real_channel_participation(peaks: Sequence[HighIntensity]):
	"""Count real participation frequency for each channel"""
	num_channels = len(peaks[0].channel_status)
	ch_freq = Counter({f'ch{i+1}': 0 for i in range(num_channels)})

	for peak in peaks:
		for i, active in enumerate(peak.channel_status):
			if active:
				ch_freq[f'ch{i+1}'] += 1

	total_events = len(peaks)
	ch_percent = {ch: count / total_events * 100 for ch, count in ch_freq.items()}

	return ch_freq, ch_percent

def build_permutation_distribution(peaks: Sequence[HighIntensity], n_perm=1000, seed=42):
	"""Build permutation-based null distribution"""
	np.random.seed(seed)
	num_channels = len(peaks[0].channel_status)
	all_channels = [f'ch{i+1}' for i in range(num_channels)]
	ch_perm_freq_list = []

	for _ in tqdm(range(n_perm), desc="Running permutations"):
		ch_freq = Counter({ch: 0 for ch in all_channels})

		for peak in peaks:
			k = sum(peak.channel_status)
			perm_channels = np.random.choice(all_channels, size=k, replace=False)
			for ch in perm_channels:
				ch_freq[ch] += 1

		ch_perm_freq_list.append(dict(ch_freq))

	return ch_perm_freq_list

def compute_p_values(real_freq: dict, perm_distributions: Sequence[dict]):
	"""Compute one-sided (right-tailed) p-values"""
	n_perm = len(perm_distributions)
	p_values = {}

	for ch in real_freq.keys():
		real = real_freq[ch]
		perm_values = [dist[ch] for dist in perm_distributions]
		p = (sum(perm >= real for perm in perm_values) + 1) / (n_perm + 1)
		p_values[ch] = p

	return p_values

def apply_fdr_correction(p_values: dict, alpha=0.05):
	"""FDR correction (Benjamini-Hochberg)"""
	ch_names = list(p_values.keys())
	pvals = [p_values[ch] for ch in ch_names]
	reject, pvals_corrected, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')

	return dict(zip(ch_names, pvals_corrected)), dict(zip(ch_names, reject))

def peak_permutation_test(csv_path: str, n_perm=1000, seed=42)-> Sequence[ChannelTestResult]:
	"""整体测试流程"""
	peaks = HighIntensity.load_sequence_from_csv(csv_path)
	real_freq, real_percent = count_real_channel_participation(peaks)
	perm_distributions = build_permutation_distribution(peaks, n_perm=n_perm, seed=seed)
	p_values = compute_p_values(real_freq, perm_distributions)
	pvals_corrected, reject = apply_fdr_correction(p_values)

	results = [
		ChannelTestResult(
			channel=ch,
			real_count=real_freq[ch],
			real_percent=real_percent[ch],
			p_value=p_values[ch],
			p_value_corrected=pvals_corrected[ch],
			significant=reject[ch]
		)
		for ch in real_freq
	]

	return results, len(peaks), perm_distributions, real_freq

if __name__ == '__main__':
	HighIntensityAnalyzer(ds_dir='out/trainingcamp-mne-april')
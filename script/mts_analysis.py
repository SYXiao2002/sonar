"""
File Name: mts_analysis.py
Author: Yixiao Shen
Date: 2025-05-28
Purpose: Multi-channel Trend Synchrony
"""
import os
from collections import Counter
from typing import Optional, Sequence

import pandas as pd

import numpy as np
from matplotlib import cm, colors, pyplot as plt
from pyparsing import annotations
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from sonar.analysis.intensity_analysis import Peak
from sonar.core.region_selector import RegionSelector
from sonar.preprocess.sv_marker import Annotation, read_annotations
from sonar.utils.topomap_plot import get_meta_data, normalize_positions, plot_anatomical_labels

def plot_ch_count_heatmap(freq_dict, label, output_dir):
	fig = plt.figure(figsize=(12, 8))
	main_ax = fig.add_subplot(111)
	main_ax.axis('off')
	os.makedirs(output_dir, exist_ok=True)

	box_width = 0.07
	box_height = 0.10

	# === 加载通道位置信息 ===
	metadata_path = "res/test/snirf_metadata.csv"
	ch_pos_l, ch_name_l = get_meta_data(metadata_path)
	ch_pos_l = normalize_positions(ch_pos_l, box_width, box_height, x_range=(0.02, 0.9), y_range=(0.05, 0.9))

	# === 频率值归一化到 0~1，用于映射颜色 ===
	values = list(freq_dict.values())
	norm = colors.Normalize(vmin=85, vmax=100)
	cmap = cm.get_cmap('gist_yarg')  # 可换为 'hot' 'viridis' 等

	for ch_idx, (x, y) in enumerate(ch_pos_l):
		x0 = x - box_width / 2
		y0 = y - box_height / 2
		ax_inset = fig.add_axes([x0, y0, box_width, box_height])
		ax_inset.set_xticks([])
		ax_inset.set_yticks([])

		# === 获取频率值和颜色 ===
		key = f'ch{ch_idx+1}'
		val = freq_dict.get(key, 0)
		color = cmap(norm(val))
		ax_inset.set_title(f'Ch{ch_idx+1}: {int(val)}%', fontsize=7, pad=2)

		# === 用颜色填满整个子图 ===
		ax_inset.set_facecolor(color)

	# === 添加颜色条 ===
	cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
	cbar=plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
	cbar.set_label('Contribution Rate', fontsize=10)

	# === 绘制背景结构等（如需） ===
	plot_anatomical_labels(plt, 2)

	plt.suptitle(f'Topomap: Channel Contribution Rate, {label}', fontsize=14)
	plt.savefig(os.path.join(output_dir, f"{label}.png"), dpi = 600)


import numpy as np
import pandas as pd
from typing import Sequence
from collections import Counter
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

# 假设 Peak 类已定义并可正常导入
# from yourmodule import Peak

def count_real_channel_participation(peaks: Sequence[Peak]):
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

def build_permutation_distribution(peaks: Sequence[Peak], n_perm=1000, seed=42):
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

def peak_permutation_test(csv_path: str, n_perm=1000, seed=42):
	"""整体测试流程"""
	peaks = Peak.load_sequence_from_csv(csv_path)
	real_freq, real_percent = count_real_channel_participation(peaks)
	perm_distributions = build_permutation_distribution(peaks, n_perm=n_perm, seed=seed)
	p_values = compute_p_values(real_freq, perm_distributions)
	pvals_corrected, reject = apply_fdr_correction(p_values)

	results = []
	for ch in real_freq.keys():
		results.append({
			'channel': ch,
			'real_count': real_freq[ch],
			'real_percent': real_percent[ch],
			'p_value': p_values[ch],
			'p_value_corrected': pvals_corrected[ch],
			'significant': reject[ch]
		})

	return pd.DataFrame(results)

# 示例使用
if __name__ == "__main__":
	csv_path = 'out/trainingcamp-pure/raw_high_intensity/HC1.csv'
	results_df = peak_permutation_test(csv_path, n_perm=1000, seed=42)
	print(results_df)

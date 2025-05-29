"""
File Name: mts_analysis.py
Author: Yixiao Shen
Date: 2025-05-28
Purpose: Multi-channel Trend Synchrony
"""


import os
from matplotlib import cm, colors, pyplot as plt
import numpy as np
import pandas as pd

from sonar.utils.topomap_plot import get_meta_data, normalize_positions, plot_anatomical_labels


def compute_channel_event_participation(peak_csv_path, bin_csv_path):
	# 读取峰值（MTS事件）数据
	df_peaks = pd.read_csv(peak_csv_path)
	events = [(start, start + dur) for start, dur in zip(df_peaks['TIME'], df_peaks['DURATION'])]

	total_events = len(events)
	if total_events == 0:
		raise ValueError("No MTS events found.")

	# 读取二值趋势矩阵
	df_bin = pd.read_csv(bin_csv_path)
	time_arr = df_bin['time'].values
	ch_cols = [col for col in df_bin.columns if col.startswith('ch')]

	# 初始化通道参与事件的次数
	ch_freq = {ch: 0 for ch in ch_cols}

	for start, end in events:
		# 获取该事件窗口的所有行
		mask = (time_arr >= start) & (time_arr <= end)
		event_data = df_bin.loc[mask, ch_cols]

		# 检查每个通道是否至少出现过1次“1”
		for ch in ch_cols:
			if event_data[ch].any():
				ch_freq[ch] += 1

	# 转为百分比
	ch_percent = {ch: freq / total_events * 100 for ch, freq in ch_freq.items()}

	return ch_percent

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




if __name__ == '__main__':
	peaks_csv_path = 'out/test_min_duation_1.6s/peaks_raw/test-sub1.csv'
	binary_csv_path = 'out/test_min_duation_1.6s/binery_trends_raw/test-sub1.csv'
	dict = compute_channel_event_participation(peaks_csv_path, binary_csv_path)
	plot_ch_count_heatmap(dict, 'test-sub1', 'out/test_min_duation_1.6s/ch_count_heatmap')
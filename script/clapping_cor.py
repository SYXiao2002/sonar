import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ==== 参数 ====
pre_sec = 20
post_sec = 20
total_win = pre_sec + post_sec
step = 1  # correlation 采样频率为 1Hz

def main(ds_dir):
	# ==== 加载 correlation 数据 ====
	ibc_df = pd.read_csv(os.path.join('out', ds_dir, 'raw_ibc/ibc.tsv'), sep='\t')
	correlation = ibc_df['correlation'].values
	time_series = ibc_df['time_sec'].values

	# ==== 加载鼓掌注释 ====
	clap_df = pd.read_csv(os.path.join('res', ds_dir, 'marker/clapping.csv'))

	# 定义鼓掌类型排序顺序和颜色映射
	label_order = ['on-clapping', 'ing-clapping', 'off-clapping']
	label_color_map = {
		'on-clapping': 'deepskyblue',
		'ing-clapping': 'orange',
		'off-clapping': 'limegreen'
	}

	# 按鼓掌类型排序
	clap_df['LABEL'] = clap_df['LABEL'].astype(str)
	clap_df['label_rank'] = clap_df['LABEL'].map({k: i for i, k in enumerate(label_order)})
	clap_df_sorted = clap_df.sort_values('label_rank').reset_index(drop=True)

	# ==== 构建 epoch ====
	epochs = []
	duration_list = []
	label_list = []

	for _, row in clap_df_sorted.iterrows():
		onset = row['TIME']
		duration = row['DURATION']
		label = row['LABEL']

		start_time = onset - pre_sec
		end_time = onset + post_sec
		start_idx = np.searchsorted(time_series, start_time)
		end_idx = np.searchsorted(time_series, end_time)

		if start_idx >= 0 and end_idx <= len(correlation) and (end_idx - start_idx == total_win):
			segment = correlation[start_idx:end_idx]
			epochs.append(segment)
			duration_list.append(duration)
			label_list.append(label)

	epochs = np.array(epochs)
	duration_list = np.array(duration_list)
	time_axis = np.arange(-pre_sec, post_sec)

	# ==== 绘图 ====
	plt.figure(figsize=(10, 6))

	# extent 的 y 范围从 0.5 到 N+0.5，确保每行居中对齐整数 y
	im = plt.imshow(
		epochs,
		aspect='auto',
		cmap='viridis',
		extent=[-pre_sec, post_sec, 0.5, len(epochs) + 0.5],
		vmin=0,
		vmax=1,
		origin='lower'
	)

	# duration 横线画在整数位置上
	for i, (dur, label) in enumerate(zip(duration_list, label_list)):
		plt.plot([0, dur], [i + 1, i + 1], color=label_color_map[label], linewidth=2)

	# Clap onset 垂直线
	plt.axvline(0, color='red', linestyle='--')

	# 图例
	legend_elements = [
		Line2D([0], [0], color='red', linestyle='--', label='Clap Onset'),
		Line2D([0], [0], color=label_color_map['off-clapping'], linewidth=2, label='off-clapping'),
		Line2D([0], [0], color=label_color_map['ing-clapping'], linewidth=2, label='ing-clapping'),
		Line2D([0], [0], color=label_color_map['on-clapping'], linewidth=2, label='on-clapping'),
	]
	plt.legend(handles=legend_elements, loc='upper right')

	# 坐标轴设置
	plt.colorbar(im, label='Inter-brain Correlation')
	plt.xlabel('Time (s) relative to clapping onset')
	plt.ylabel('Clapping event index (sorted by type)')
	plt.yticks(ticks=np.arange(1, len(epochs) + 1))  # 设置 y tick 为整数
	plt.title('Event-locked Inter-brain Correlation with Clapping Type and Duration, n={}'.format(len(epochs)))
	plt.tight_layout()
	plt.savefig(os.path.join('out', ds_dir, 'event-locked-ibc-with-clap.png'), dpi=600)


if __name__ == "__main__":
	main(ds_dir='trainingcamp-nirspark')
	main(ds_dir='trainingcamp-homer3')
	main(ds_dir='trainingcamp-mne-april')
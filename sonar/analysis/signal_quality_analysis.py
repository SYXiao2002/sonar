import os
from itertools import compress

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mne.preprocessing.nirs import optical_density

from mne_nirs.preprocessing import scalp_coupling_index_windowed
import pandas as pd
from sonar.preprocess.mne_converter import crop_data, get_trigger_times, read_snirf
from sonar.preprocess.snirf_metadata import load_idx_remap_dict

# matplotlib.use('Agg')  # Use non-interactive backend

def plot_timechannel_quality_metric(raw, scores, times, threshold=0.1, title=None, channel_dict=None):

	# 去重，只保留每个物理通道的第一个波长
	seen = set()
	unique_indices = []
	unique_ch_names = []

	for i, ch_name in enumerate(raw.ch_names):
		physical_name = ch_name.split()[0]  # 去掉波长，只保留物理通道名
		if physical_name not in seen:
			seen.add(physical_name)
			unique_indices.append(i)
			unique_ch_names.append(physical_name)

	if channel_dict is not None:
		unique_ch_names = [channel_dict[ch] for ch in unique_ch_names]

	# 将通道名转换为 numpy 数组以便排序
	unique_ch_names = np.array(unique_ch_names)

	# 获取排序后的索引（按通道名字母顺序）
	sort_idx = np.argsort(unique_ch_names)

	# 应用排序
	ch_names = unique_ch_names[sort_idx].tolist()
	ch_names = [f'CH{ch}'for ch in ch_names]
	scores = scores[sort_idx, :]


	cols = [np.round(t[0]) for t in times]

	if title is None:
		title = "Automated noisy channel detection: fNIRS"

	data_to_plot = pd.DataFrame(
		data=scores,
		columns=pd.Index(cols, name="Time (s)"),
		index=pd.Index(ch_names, name="Channel"),
	)

	n_chans = len(ch_names)
	vsize = 0.2 * n_chans

	# First, plot the "raw" scores.
	fig, ax = plt.subplots(1, 2, figsize=(20, vsize), layout="constrained")
	fig.suptitle(title, fontsize=16, fontweight="bold")
	sns.heatmap(
		data=data_to_plot,
		cmap="Reds_r",
		vmin=0,
		vmax=1,
		cbar_kws=dict(label="Score"),
		ax=ax[0],
	)
	[
		ax[0].axvline(x, ls="dashed", lw=0.25, dashes=(25, 15), color="gray")
		for x in range(1, len(times))
	]
	ax[0].set_title("All Scores", fontweight="bold")
	markbad(raw, ax[0], ch_names=ch_names, channel_dict=channel_dict)

	# Now, adjust the color range to highlight segments that exceeded the
	# limit.

	data_to_plot = pd.DataFrame(
		data=scores > threshold,
		columns=pd.Index(cols, name="Time (s)"),
		index=pd.Index(ch_names, name="Channel"),
	)
	sns.heatmap(
		data=data_to_plot,
		vmin=0,
		vmax=1,
		cmap="Reds_r",
		cbar_kws=dict(label="Score"),
		ax=ax[1],
	)
	[
		ax[1].axvline(x, ls="dashed", lw=0.25, dashes=(25, 15), color="gray")
		for x in range(1, len(times))
	]
	ax[1].set_title(f"Scores < {threshold}", fontweight="bold")
	markbad(raw, ax[1], ch_names=ch_names, channel_dict=channel_dict)

	return fig


def markbad(raw, ax, ch_names, channel_dict=None):
	# 取 raw.info['bads'] 里所有坏通道的物理名
	bad_phys_names = [ch.split()[0] for ch in raw.info['bads']]

	# 如果使用了 channel_dict 重命名，需要映射坏通道名
	if channel_dict is not None:
		bad_phys_names = [channel_dict[ch] for ch in bad_phys_names if ch in channel_dict]

	# 最终热图中显示的是 'CHxx' 格式，所以也加上前缀
	bad_chn_formatted = [f'CH{ch}' for ch in bad_phys_names]

	# 遍历热图用的通道名，画线
	for i, ch in enumerate(ch_names):
		if ch in bad_chn_formatted:
			ax.axhline(i + 0.5, ls="solid", lw=2, color="black")

	return ax

def main(ds_dir, first_trigger, last_trigger, shift_seconds):
	snirf_dir = os.path.join('res', ds_dir, 'snirf')
	snirf_file_l = sorted([
		os.path.join(snirf_dir, f) for f in os.listdir(snirf_dir) if f.endswith('.snirf')
	])
	out_dir = os.path.join('out', ds_dir, 'signal_quaility')
	metadata_path = os.path.join('res', ds_dir, 'snirf',  'snirf_metadata.csv')
	channel_dict = load_idx_remap_dict(metadata_path)

	for snirf_file in snirf_file_l:
		sub_label = os.path.splitext(os.path.basename(snirf_file))[0]
		sub_folder = os.path.join(out_dir, sub_label)
		os.makedirs(sub_folder, exist_ok=True)

		raw_intensity = read_snirf(snirf_file)
		raw_intensity.load_data().resample(4.0, npad="auto")
		raw_od: mne.io.Raw = optical_density(raw_intensity)


		# Define the roi here!!!
		if first_trigger is not None and last_trigger is not None:
			start_time, _ = get_trigger_times(raw_od, first_trigger) 
			_, end_time = get_trigger_times(raw_od, last_trigger)
			raw_od = crop_data(raw_od, start_time, end_time)

		sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)
		# fig, ax = plt.subplots()
		# ax.hist(sci)
		# ax.set(xlabel="Scalp Coupling Index", ylabel="Count", xlim=[0, 1])
		# plt.savefig(os.path.join(sub_folder, 'sci_cropped.png'))

		raw_od.info["bads"] = list(compress(raw_od.ch_names, sci < 0.8))
		print(raw_od.info["bads"])
		
		# raw_od.plot_sensors()
		# plt.savefig(os.path.join(sub_folder, 'sensors.png'))

		time_window = 60
		_, scores, times = scalp_coupling_index_windowed(raw_od, time_window=time_window)
		times = [(start + shift_seconds, end + shift_seconds) for (start, end) in times]
		plot_timechannel_quality_metric(
			raw_od,
			scores,
			times,
			threshold=0.8,
			title=f"Scalp Coupling Index (SCI): {sub_label}, Time Window = {time_window}s",
			channel_dict=channel_dict
		)
		plt.savefig(os.path.join(sub_folder, 'sci_sliding.png'))

		# raw_od, scores, times = peak_power(raw_od, time_window=10)
		# times = [(start + shift_seconds, end + shift_seconds) for (start, end) in times]
		# plot_timechannel_quality_metric(
		# 	raw_od, scores, times, threshold=0.1, title="Peak Power Quality Evaluation"
		# )
		# plt.savefig(os.path.join(sub_folder, 'peak_power_10s.png'))

if __name__ == '__main__':
	main(ds_dir='trainingcamp-mne-april', first_trigger=9, last_trigger=19, shift_seconds=2222)
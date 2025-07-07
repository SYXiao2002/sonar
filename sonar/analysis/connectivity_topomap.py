# topomap read from metadata, DO NOT forget to remap to W_channel. Also, nose is up, left hemisphere is left!!!
# single brain, multiple channel, long period correlation
# 假设48CH，使用两两pearson相关，一共计算[C48, 2]=1128次，然后保存结果为csv (二维矩阵，48行，48列，对角线上为0，其他为pearson相关系数)
# 第一种. 从csv中读取结果，然后apply Louvain community detection，尝试分割脑区
# 第二张. 从csv中读取结果，然后筛选出 top-K 强连接，只显示有意义的边，避免信息过载。
import os
from typing import Dict, Optional, Sequence

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import community.community_louvain as community_louvain
import networkx as nx

from sonar.core.color import get_color_from_label
from sonar.core.dataset_loader import DatasetLoader, get_dataset
from sonar.preprocess.snirf_metadata import get_metadata_dict, load_idx_remap_dict, normalize_metadata_pos_dict
from sonar.preprocess.sv_marker import Annotation
from sonar.utils.topomap_plot import plot_anatomical_labels


class ConnectivityTopomap():
	def __init__(
		self,
		output_dir: Optional[str],
		dataset: DatasetLoader,
		annotations: Optional[Sequence[Annotation]] = None,
		metadata_path: Optional[str] = None,
		debug: bool = False,
	):
		self.output_dir = output_dir
		self.dataset = dataset
		self.annotations = annotations
		self.debug = debug
		self.metadata_path = metadata_path

		os.makedirs(self.output_dir, exist_ok=True)

		# Default region_selector: full duration
		if self.annotations is None:
			self.annotations = [Annotation(
				start=self.dataset.raw_time[0],
				duration=self.dataset.raw_time[-1] - self.dataset.raw_time[0],
				value=0,
				label='ALL'
			)]

		# Store computed results per subject, outer dict: annotation_idx, inner dict: subject_idx
		self._computed_connectivity: Dict[int, Dict[int, np.ndarray]] = {}
		self._computed_community_partitions: Dict[int, Dict[int, np.ndarray]] = {}

		self._calculate()
		self._save()

	def _calculate(self):
		self._cal_connectivity()
		self._cal_Louvain_community_detection()

	def _cal_connectivity(self):
		time = np.array(self.dataset.raw_time)

		for a_idx, a in enumerate(self.annotations):
			start_t = a.start
			end_t = a.start + a.duration
			mask = (time >= start_t) & (time <= end_t)
			selected_idx = np.where(mask)[0]

			if self.debug:
				print(f"[DEBUG] Event {a_idx}: {start_t}s ~ {end_t}s → {len(selected_idx)} samples")

			# 初始化每个 annotation 的存储字典
			self._computed_connectivity[a_idx] = {}

			# 遍历每个被试
			for sub_idx, sub_data in tqdm(
				enumerate(self.dataset.data_l),
				total=len(self.dataset.data_l),
				desc=f"Computing correlations (event {a_idx})"
			):
				sub_label = self.dataset.label_l[sub_idx]
				data = np.stack(sub_data)  # shape: (n_channels, n_timepoints)

				# 取该事件窗口内数据
				data_event = data[:, selected_idx]

				# 计算通道间相关性
				corr_matrix = np.corrcoef(data_event)
				np.fill_diagonal(corr_matrix, 0)

				# 存入双层字典
				self._computed_connectivity[a_idx][sub_idx] = corr_matrix

				if self.debug:
					print(f"[DEBUG] Event {a.label}-{a.value} | {sub_label}: corr shape = {corr_matrix.shape}")

	def _cal_Louvain_community_detection(self):
		"""
		Calculate Louvain community detection for all annotations and subjects.
		Store the partition results to self._computed_community_partitions.
		"""
		self._computed_community_partitions = {}

		for a_idx, subj_dict in self._computed_connectivity.items():
			self._computed_community_partitions[a_idx] = {}

			for sub_idx, corr_matrix in subj_dict.items():
				channel_labels = self.dataset.ch_l

				# Only keep positively correlated edges above threshold
				threshold = np.quantile(corr_matrix[corr_matrix > 0], 0.75)
				adj = np.where((corr_matrix > 0) & (corr_matrix >= threshold), corr_matrix, 0)
				G = nx.from_numpy_array(adj)
				label_map = dict(enumerate(channel_labels))

				# Run Louvain
				partition = community_louvain.best_partition(G, weight='weight', random_state=42)


				# Stabilize community id by sorting node names within each community
				from collections import defaultdict

				inv_partition = defaultdict(list)
				for node_idx, comm_id in partition.items():
					ch_name = label_map[node_idx]
					inv_partition[comm_id].append(ch_name)

				# Sort community groups by sorted names for stable ID assignment
				sorted_groups = sorted(inv_partition.values(), key=lambda x: sorted(x))

				# Assign stable community IDs, skip groups with <=3 members
				partition_named = {}
				stable_id = 1
				for group in sorted_groups:
					if len(group) <= 3:
						for ch_name in group:
							partition_named[ch_name] = 0
					for ch_name in group:
						partition_named[ch_name] = stable_id
					stable_id += 1

				self._computed_community_partitions[a_idx][sub_idx] = partition_named

	def _save(self):
		self._save_connectivity_to_csv()
		self._plot_partitions()

	def _save_connectivity_to_csv(self):
		print("Saving connectivity matrices to CSV...")

		channel_labels = self.dataset.ch_l
		n_channels = len(self.dataset.data_l[0])

		if not channel_labels or len(channel_labels) != n_channels:
			channel_labels = [f'ch{i}' for i in range(n_channels)]

		for a_idx, subj_dict in self._computed_connectivity.items():
			for sub_idx, corr_matrix in subj_dict.items():
				a_label = f'{self.annotations[a_idx].label}-{self.annotations[a_idx].value}'
				sub_label = self.dataset.label_l[sub_idx]
				filename = f"corr_matrix_{a_label}_{sub_label}.csv"
				save_path = os.path.join(self.output_dir, 'raw_connectivity', sub_label, filename)

				os.makedirs(os.path.dirname(save_path), exist_ok=True)

				df = pd.DataFrame(corr_matrix, index=channel_labels, columns=channel_labels)
				df.to_csv(save_path)

				if self.debug:
					print(f"[DEBUG] Saved: {save_path}")

	def _plot_partitions(self, box_width=0.07, box_height=0.10):
		channel_dict = load_idx_remap_dict(self.metadata_path)
		for a_idx, subj_dict in self._computed_community_partitions.items():
			a_label = f'{self.annotations[a_idx].label}-{self.annotations[a_idx].value}'

			for sub_idx, partition in subj_dict.items():
				sub_label = self.dataset.label_l[sub_idx]

				metadata_path = os.path.join(self.metadata_path)
				metadata_dict, _ = get_metadata_dict(metadata_path)
				metadata_dict = normalize_metadata_pos_dict(metadata_dict, box_width, box_height)

				fig = plt.figure(figsize=(12, 8))
				plt.suptitle(f'Louvain Community: {a_label}, {sub_label}', fontsize=14)

				for key, value in partition.items():
					ch_name = key.split()[0]
					assert ch_name in metadata_dict, f"Channel '{ch_name}' not found in partition: {metadata_dict}"
					pos = metadata_dict[ch_name].pos
					if value == 0:
						color = 'white'
					else:
						color = get_color_from_label(str(value))

					# Get subplot position in normalized coordinates
					x0 = pos[0] - box_width / 2
					y0 = pos[1] - box_height / 2

					ax_inset = fig.add_axes([x0, y0, box_width, box_height])
					ax_inset.set_facecolor(color)
					ax_inset.set_xticks([])
					ax_inset.set_yticks([])
					ax_inset.set_xticklabels([])
					ax_inset.set_yticklabels([])
					ax_inset.set_title(f'CH{channel_dict[ch_name]}', fontsize=7, color='black')

					ax_inset.text(0.5, 0.5, ch_name, ha='center', va='center', fontsize=7, color='white')

				plot_anatomical_labels(plt, template_idx=1)

				out_dir = os.path.join(self.output_dir, 'fig_community_partitions', sub_label)	
				os.makedirs(out_dir, exist_ok=True)

				save_path = os.path.join(out_dir, f'community_partition_{sub_label}_{a_label}.png')
				plt.savefig(save_path, dpi=600)

def main(ds_dir, load_cache, debug, marker_file):
	ds, annotations = get_dataset(ds_dir=os.path.join('res', ds_dir), load_cache=load_cache, marker_file=marker_file)
	metadata_path = os.path.join('res', ds_dir, 'snirf',  'snirf_metadata.csv')
	vc = ConnectivityTopomap(
		output_dir = os.path.join('out', ds_dir),
		dataset = ds,
		debug = debug,
		annotations=annotations,
		metadata_path=metadata_path
	)	

if __name__ == '__main__':
	# main(ds_dir='test', load_cache=True, debug=True, marker_file=None)
	main(ds_dir='trainingcamp-mne-april', load_cache=True, debug=True, marker_file=None)
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon

def load_data(ds_dir='trainingcamp-nirspark'):
	# Load correlation and clapping annotation data
	ibc_df = pd.read_csv(os.path.join('out', ds_dir, 'raw_ibc/ibc.tsv'), sep='\t')
	correlation = ibc_df['correlation'].values
	time_series = ibc_df['time_sec'].values

	clap_df = pd.read_csv(os.path.join('res', ds_dir, 'marker/clapping.csv'))
	return correlation, time_series, clap_df

def extract_epochs(correlation, time_series, clap_df, pre_sec, post_sec):
	total_win = pre_sec + post_sec

	# Define clap label order and sort
	label_order = ['on-clapping', 'ing-clapping', 'off-clapping']
	clap_df['LABEL'] = clap_df['LABEL'].astype(str)
	clap_df['label_rank'] = clap_df['LABEL'].map({k: i for i, k in enumerate(label_order)})
	clap_df_sorted = clap_df.sort_values('label_rank').reset_index(drop=True)

	epochs = []
	for _, row in clap_df_sorted.iterrows():
		onset = row['TIME']
		start_time = onset - pre_sec
		end_time = onset + post_sec
		start_idx = np.searchsorted(time_series, start_time)
		end_idx = np.searchsorted(time_series, end_time)
		if start_idx >= 0 and end_idx <= len(correlation) and (end_idx - start_idx == total_win):
			segment = correlation[start_idx:end_idx]
			epochs.append(segment)

	epochs = np.array(epochs)
	time_axis = np.arange(-pre_sec, post_sec)
	return epochs, time_axis, clap_df_sorted

def compute_pre_post_means(epochs, time_axis, pre_sec=5, post_sec=20):
	pre_mask = (time_axis >= -pre_sec) & (time_axis < 0)
	post_mask = (time_axis >= 0) & (time_axis < post_sec)

	pre_vals = epochs[:, pre_mask].mean(axis=1)
	post_vals = epochs[:, post_mask].mean(axis=1)
	return pre_vals, post_vals

def permutation_test(correlation, time_series, n_events, pre_sec, post_sec, total_win, true_diff, n_perm=100000):
	min_time = time_series[0] + pre_sec
	max_time = time_series[-1] - post_sec

	rand_diffs = []
	for _ in range(n_perm):
		rand_onsets = np.random.uniform(min_time, max_time, size=n_events)

		rand_epochs = []
		for onset in rand_onsets:
			start_time = onset - pre_sec
			end_time = onset + post_sec
			start_idx = np.searchsorted(time_series, start_time)
			end_idx = np.searchsorted(time_series, end_time)
			if start_idx >= 0 and end_idx <= len(correlation) and (end_idx - start_idx == total_win):
				segment = correlation[start_idx:end_idx]
				rand_epochs.append(segment)

		if len(rand_epochs) < n_events:
			continue

		rand_epochs = np.array(rand_epochs)
		time_axis = np.arange(-pre_sec, post_sec)
		pre_mask = (time_axis >= -pre_sec) & (time_axis < 0)
		post_mask = (time_axis >= 0) & (time_axis < post_sec)

		pre_means = rand_epochs[:, pre_mask].mean(axis=1)
		post_means = rand_epochs[:, post_mask].mean(axis=1)

		rand_diff = post_means.mean() - pre_means.mean()
		rand_diffs.append(rand_diff)

	rand_diffs = np.array(rand_diffs)
	p_value = (np.sum(np.abs(rand_diffs) >= np.abs(true_diff)) + 1) / (len(rand_diffs) + 1)
	return rand_diffs, p_value

def main(ds_dir, n_perm=10):
	pre_sec = 20
	post_sec = 20
	total_win = pre_sec + post_sec

	correlation, time_series, clap_df = load_data(ds_dir=ds_dir)
	epochs, time_axis, clap_df_sorted = extract_epochs(correlation, time_series, clap_df, pre_sec, post_sec)
	pre_vals, post_vals = compute_pre_post_means(epochs, time_axis, pre_sec, post_sec)

	# Paired t-test
	t_stat, p_t = ttest_rel(post_vals, pre_vals)
	# Wilcoxon signed-rank test
	w_stat, p_w = wilcoxon(post_vals, pre_vals)
	# Permutation test
	true_diff = post_vals.mean() - pre_vals.mean()
	rand_diffs, p_perm = permutation_test(correlation, time_series, len(epochs), pre_sec, post_sec, total_win, true_diff, n_perm)

	print(f"Paired t-test: t = {t_stat:.3f}, p = {p_t:.4f}")
	print(f"Wilcoxon test: W = {w_stat:.3f}, p = {p_w:.4f}")
	print(f"Permutation test: mean diff = {true_diff:.4f}, p = {p_perm:.4f}")
	print(f"Permutation test random diffs count: {len(rand_diffs)}")
	print(f"Permutation test random diffs mean: {np.mean(rand_diffs):.4f}, std: {np.std(rand_diffs):.4f}")

	# 先定义颜色映射，区分三种clapping类型
	color_map = {
		'off-clapping': 'tab:green',
		'ing-clapping': 'tab:orange',
		'on-clapping': 'tab:blue',
	}

	# 假设clap_df_sorted是extract_epochs中排序后的clap_df
	# 你要保证clap_df_sorted['LABEL']和epochs顺序对应
	clapping_types = clap_df_sorted['LABEL'].values  # array/list

	plt.figure(figsize=(6, 4))

	for i in range(len(pre_vals)):
		ctype = clapping_types[i]
		plt.plot([0, 1], [pre_vals[i], post_vals[i]], marker='o',
				color=color_map.get(ctype, 'gray'), alpha=0.8,
				label=ctype if i == 0 else None)  # 只在第一个对应类型加label，避免重复图例

	# 添加图例（避免重复标签）
	handles = []
	labels = []
	for ctype, color in color_map.items():
		handles.append(plt.Line2D([0], [0], color=color, marker='o', linestyle='-'))
		labels.append(ctype)
	plt.legend(handles, labels, title='Clapping Type')

	plt.xticks([0, 1], ['Pre-clap', 'Post-clap'])
	plt.ylabel('Mean Inter-brain Correlation')
	plt.title(f'Pre vs Post Clapping IBC, n={len(epochs)}, t = {t_stat:.3f}, p = {p_t:.4f}')
	plt.grid(True, linestyle='--', alpha=0.3)
	plt.tight_layout()
	plt.savefig(os.path.join('out', ds_dir, 'pre-post-ibc.png'), dpi=600)

	# Plot permutation null distribution
	plt.figure(figsize=(6, 4))
	plt.hist(rand_diffs, bins=30, alpha=0.7, label='Random Differences')
	plt.axvline(true_diff, color='red', linestyle='--', label='True Difference')
	plt.xlabel('Mean difference (Post - Pre)')
	plt.ylabel('Frequency')
	plt.legend()
	plt.title(f'Permutation Test Null Distribution, shuffle={len(rand_diffs)}, p = {p_perm:.4f}')
	plt.tight_layout()
	plt.savefig(os.path.join('out', ds_dir, 'perm-test-null.png'), dpi=600)

	# Save results to JSON
	result_dict = {
		"ds_dir": ds_dir,
		"n_epochs": len(epochs),
		"t_test": {
			"t_stat": round(t_stat, 4),
			"p_value": round(p_t, 6)
		},
		"wilcoxon": {
			"w_stat": round(w_stat, 4),
			"p_value": round(p_w, 6)
		},
		"permutation": {
			"true_diff": round(true_diff, 6),
			"p_value": round(p_perm, 6),
			"rand_diff_mean": round(np.mean(rand_diffs), 6),
			"rand_diff_std": round(np.std(rand_diffs), 6),
			"n_valid_samples": len(rand_diffs)
		}
	}

	json_path = os.path.join('out', ds_dir, 'pre_post_ibc_stats.json')
	with open(json_path, 'w') as f:
		json.dump(result_dict, f, indent=2)

	print(f"Saved JSON results to {json_path}")

if __name__ == "__main__":
	main(ds_dir='trainingcamp-nirspark')
	main(ds_dir='trainingcamp-homer3')
	main(ds_dir='trainingcamp-mne-april')

import os
import re
import textwrap
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as patches

from sonar.core.color import get_color_from_label
from sonar.preprocess.snirf_metadata import get_metadata_dict, normalize_metadata_pos_dict
from sonar.utils.topomap_plot import plot_anatomical_labels

# from https://github.com/matplotlib/matplotlib/issues/6321#issuecomment-555587961
def annotate_yrange(ymin, ymax,
					label=None,
					offset=-0.1,
					width=-0.1,
					ax=None,
					patch_kwargs={'facecolor':'gray'},
					line_kwargs={'color':'black'},
					text_kwargs={'rotation':'vertical'}
):
	import matplotlib.pyplot as plt
	import matplotlib.transforms as transforms

	from matplotlib.patches import Rectangle
	from matplotlib.lines import Line2D
	if ax is None:
		ax = plt.gca()

	# x-coordinates in axis coordinates,
	# y coordinates in data coordinates
	trans = transforms.blended_transform_factory(
		ax.transAxes, ax.transData)

	# a bar indicting the range of values
	rect = Rectangle((offset, ymin), width=width, height=ymax-ymin,
					 transform=trans, clip_on=False, **patch_kwargs)
	ax.add_patch(rect)

	# delimiters at the start and end of the range mimicking ticks
	min_delimiter = Line2D((offset+width, offset), (ymin, ymin),
						   transform=trans, clip_on=False, **line_kwargs)
	max_delimiter = Line2D((offset+width, offset), (ymax, ymax),
						   transform=trans, clip_on=False, **line_kwargs)
	ax.add_artist(min_delimiter)
	ax.add_artist(max_delimiter)

	# label
	if label:
		x = offset + 0.5 * width
		y = ymin + 0.5 * (ymax - ymin)
		# we need to fix the alignment as otherwise our choice of x
		# and y leads to unexpected results;
		# e.g. 'right' does not align with the minimum_delimiter
		ax.text(x, y, label,
				horizontalalignment='center', verticalalignment='center',
				clip_on=False, transform=trans, **text_kwargs)

def map_channel_idx_to_y_axis(df: pd.DataFrame, tol: float = 1e-6) -> dict:
	"""
	Map 2D topomap coordinates to 1D y-axis position using channel_idx as key.
	Parameters:
		df (pd.DataFrame): Input DataFrame with 'x', 'y', and 'channel_idx' columns
		tol (float): Tolerance to detect midline channels
	Returns:
		dict: Mapping from channel_idx to y_axis_pos
	"""
	# Compute median x to define midline
	x_median = df['x'].median()

	# Label channel position (HC layout, down is nose, left is right hemisphere)
	def classify_channel(x):
		if abs(x - x_median) < tol:
			return 'midline'
		elif x < x_median:
			return 'right'
		else:
			return 'left'

	df = df.copy()
	df['label'] = df['x'].apply(classify_channel)

	# Sort by position rules
	right = df[df['label'] == 'right'].sort_values(by=['y', 'x'], ascending=[False, False])
	left = df[df['label'] == 'left'].sort_values(by=['y', 'x'], ascending=[True, False])
	midline = df[df['label'] == 'midline'].sort_values(by='x')

	# Concatenate in y-axis order
	mapped = pd.concat([left, midline, right], ignore_index=True)
	mapped['y_axis_pos'] = range(len(mapped))

	# Create mapping dict
	return dict(zip(mapped['channel_idx'], mapped['y_axis_pos'])), mapped

# for HC only
def parse_region_csv(region_csv_path):
	df = pd.read_csv(region_csv_path)
	for col in df.select_dtypes(include='object').columns:
		df[col].fillna('', inplace=True)

	region_dict = {}
	current_ch = ''
	current_entries = []

	# 获取第二列的列名
	region_column_name = df.columns[1]

	for _, row in df.iterrows():
		label = str(row.iloc[0]).strip()
		region = str(row.iloc[1]).strip()
		percentage = row['Percentage']

		if label.startswith('CH'):  # new channel block
			if current_ch and current_entries:
				# sort and save previous channel
				sorted_entries = sorted(current_entries, key=lambda x: x[1], reverse=True)
				region_dict[current_ch] = sorted_entries
			assert '(' in label and ')' in label, f"Channel label format error: {label}"
			current_ch = label.split('(')[1].split(')')[0].strip().replace('-', '_') # CH1 (S1-D1) -> S1_D1
			current_entries = []

		if region and isinstance(percentage, (float, int)):
			current_entries.append((region, float(percentage)))

	# Save last block
	if current_ch and current_entries:
		sorted_entries = sorted(current_entries, key=lambda x: x[1], reverse=True)
		region_dict[current_ch] = sorted_entries

	return region_dict, region_column_name

def get_region_colors(region_dict, colormap=plt.cm.tab20):
	all_regions = set()
	for region_list in region_dict.values():
		for region, _ in region_list:
			all_regions.add(region)

	color_pool = colormap.colors
	region_list_sorted = sorted(list(all_regions))  # consistent ordering
	region_colors = {
		region: color_pool[i % len(color_pool)]
		for i, region in enumerate(region_list_sorted)
	}
	return region_colors


def plot_brain_region_labels(
	fig,
	metadata_dict,
	region_dict,
	box_width,
	box_height,
	br_thr,
	fontsize,
	n_subregions=3,
	use_global_colors: bool = True
):
	if use_global_colors:
		region_colors_global = get_region_colors(region_dict)

	for ch_name, ch_metadata in metadata_dict.items():
		ch_idx = ch_metadata.idx
		x = ch_metadata.pos[0]
		y = ch_metadata.pos[1]

		x0 = x - box_width / 2
		y0 = y - box_height / 2
		ax_inset = fig.add_axes([x0, y0, box_width, box_height])
		ax_inset.set_xticks([])
		ax_inset.set_yticks([])
		# ax_inset.set_title(f'Ch{ch_idx}', fontsize=7, pad=2)
		ax_inset.set_title(f'{ch_name}', fontsize=7, pad=2)

		assert ch_name in region_dict, f"{ch_name} not found in region_dict{region_dict}"
		top_regions = region_dict[ch_name][:n_subregions]

		if use_global_colors:
			region_colors = region_colors_global
		else:
			# Per-channel color assignment
			region_colors = {}
			color_pool = plt.cm.tab20.colors
			for idx, (region, _) in enumerate(top_regions):
				if region not in region_colors:
					region_colors[region] = color_pool[idx % len(color_pool)]

		# Draw color blocks from top to bottom
		pathch_y_start = 1
		for region, percent in top_regions:
			if percent < br_thr / 100:
				continue

			height = percent
			ax_inset.add_patch(patches.Rectangle(
				(0, pathch_y_start - height),
				1, height,
				transform=ax_inset.transAxes,
				color=region_colors[region],
				lw=0, edgecolor='none'
			))

			text = f"({percent:.0%}) {region}"
			wrapped_text = textwrap.fill(text, width=25)

			ax_inset.text(
				0.05, pathch_y_start - 0.05,
				wrapped_text,
				fontsize=fontsize,
				va='top',
				ha='left',
				wrap=False
			)
			pathch_y_start -= height

def topomap_brain_region(region_csv_path, br_thr=15, debug=False, box_width = 0.07, box_height = 0.10, use_global_colors: bool = True, reverse=True):
	fig = plt.figure(figsize=(12, 8))
	main_ax = fig.add_subplot(111)
	main_ax.axis('off')

	dir_path = os.path.dirname(region_csv_path)

	metadata_path = os.path.join(dir_path, 'snirf_metadata.csv')
	metadata_dict, _ = get_metadata_dict(metadata_path)
	metadata_dict = normalize_metadata_pos_dict(metadata_dict, box_width, box_height, reverse=reverse)

	region_dict, classifier_type = parse_region_csv(region_csv_path)

	plot_brain_region_labels(
		fig,
		metadata_dict,
		region_dict,
		box_width,
		box_height,
		br_thr,
		fontsize=5,
		use_global_colors=use_global_colors
	)
	if reverse:
		plot_anatomical_labels(plt, template_idx=1)	
	else:
		plot_anatomical_labels(plt, template_idx=0)
	fig.suptitle(f"Topomap: Brain Regions >= {br_thr}%, from {classifier_type}", fontsize=14)
	suffix = "global" if use_global_colors else "local"
	if debug:
		plt.show()
	else:
		plt.savefig(os.path.join(dir_path, f"{classifier_type}_{suffix}.png"), dpi = 600)

if __name__ == "__main__":
	debug = False
	use_global_colors = True
	# topomap_brain_region('res/brain-region-sd/PFC+MOTOR/PFC+MOTOR_MRIcro.csv', debug=debug, box_width = 0.05, box_height = 0.07)



	# PFC
	csv_l = [
		# 'res/brain-region-sd/temp/Brodmann(MRIcro).csv',
		'res/brain-region-sd/temp/Brodmann(MRIcro).csv',
		# 'res/brain-region-sd/PFC/AAL.csv', 
		# 'res/brain-region-sd/PFC/Brodmann(MRIcro).csv',
		# 'res/brain-region-sd/PFC/LPBA40.csv',
		# 'res/brain-region-sd/PFC/Talairach.csv',
	]
	for csv in csv_l:
		topomap_brain_region(csv, debug=debug, use_global_colors=use_global_colors, reverse=True)
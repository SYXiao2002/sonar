import os
import textwrap
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib.patches as patches
import textwrap

from sonar.utils.topomap_plot import get_meta_data, normalize_positions, plot_anatomical_labels

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

	# Label channel position
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
	right = df[df['label'] == 'right'].sort_values(by=['y', 'x'], ascending=[True, False])
	left = df[df['label'] == 'left'].sort_values(by=['y', 'x'], ascending=[True, True])
	midline = df[df['label'] == 'midline'].sort_values(by='x')

	# Concatenate in y-axis order
	mapped = pd.concat([left, midline, right], ignore_index=True)
	mapped['y_axis_pos'] = range(len(mapped))

	# Create mapping dict
	return dict(zip(mapped['channel_idx'], mapped['y_axis_pos'])), mapped

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
			current_ch = label.split()[0]  # CH1 (S1-D1) -> CH1
			current_entries = []

		if region and isinstance(percentage, (float, int)):
			current_entries.append((region, float(percentage)))

	# Save last block
	if current_ch and current_entries:
		sorted_entries = sorted(current_entries, key=lambda x: x[1], reverse=True)
		region_dict[current_ch] = sorted_entries

	return region_dict, region_column_name

import matplotlib.patches as patches
import textwrap

def plot_brain_region_labels(fig, ch_pos_l, region_dict, box_width, box_height, br_thr, fontsize):

	# Step 3: Add label boxes to the figure
	for ch_idx, (x, y) in enumerate(ch_pos_l):
		x0 = x - box_width / 2
		y0 = y - box_height / 2
		ax_inset = fig.add_axes([x0, y0, box_width, box_height])
		ax_inset.set_xticks([])
		ax_inset.set_yticks([])

		ch_key = f"CH{ch_idx+1}"
		ax_inset.set_title(f'Ch{ch_idx+1}', fontsize=7, pad=2)

		if ch_key in region_dict:
			max_lines = 10
			top_regions = region_dict[ch_key][:max_lines]

			# Color map for regions
			region_colors = {}
			color_pool = plt.cm.tab20.colors  # 20 distinct colors

			for idx, (region, _) in enumerate(top_regions):
				if region not in region_colors:
					region_colors[region] = color_pool[idx % len(color_pool)]

			# Draw color blocks from top to bottom
			pathch_y_start = 1  # Start from the top
			bar_height = 0.3  # Adjust for space

			for region, percent in top_regions:
				if percent < br_thr/100:  # Skip regions with percentage less than threshold
					continue

				height = percent  # Height proportional to the percentage
				ax_inset.add_patch(patches.Rectangle(
					(0, pathch_y_start - height),  # Position from the top, reducing y_start
					1, height,  # Full width, proportional height
					transform=ax_inset.transAxes,
					color=region_colors[region],  # Color corresponding to the region
					lw=0, edgecolor='none'
				))

				# Prepare the text with wrapping
				text = f"({percent:.0%}) {region}"
				wrapped_text = textwrap.fill(text, width=25)  # Wrap the text at 25 characters

				# Add wrapped text in the left and top position of the color block
				ax_inset.text(
					0.05, pathch_y_start-0.05,  # Position text to the left and top
					wrapped_text,
					fontsize=fontsize,
					va='top',  # Vertically aligned to the top of the block
					ha='left',  # Horizontally aligned to the left of the block
					wrap=False  # Don't wrap again, as we've already wrapped text
				)
				pathch_y_start -= height  # Update y_start for the next block

		else:
			ax_inset.text(0.5, 0.5, "No data", fontsize=fontsize, va='center', ha='center')
	

def topomap_brain_region(region_csv_path, br_thr=15, debug=False, box_width = 0.07, box_height = 0.10):
	fig = plt.figure(figsize=(12, 8))
	main_ax = fig.add_subplot(111)
	main_ax.axis('off')

	# 
	dir_path = os.path.dirname(region_csv_path)

	metadata_path = os.path.join(dir_path, 'snirf_metadata.csv')
	ch_pos_l, ch_name_l = get_meta_data(metadata_path)
	ch_pos_l = normalize_positions(ch_pos_l, box_width, box_height)

	region_dict, region_column_name = parse_region_csv(region_csv_path)

	plot_brain_region_labels(
		fig,
		ch_pos_l,
		region_dict,
		box_width,
		box_height,
		br_thr,
		fontsize=5,
	)
	plot_anatomical_labels(plt)	
	fig.suptitle(f"Topomap: Brain Regions >= {br_thr}%, from {region_column_name}", fontsize=14)
	if debug:
		plt.show()
	else:
		plt.savefig(os.path.join(dir_path, f"{region_column_name}.png"), dpi = 1200)

if __name__ == "__main__":
	# region_csv_l = [
	# 	'res/brain-region-sd/PFC/Brodmann(MRIcro).csv',
	# ]
	debug=False

	# for region_csv in region_csv_l:
	# 	topomap_brain_region(region_csv, debug=debug)

	topomap_brain_region('res/brain-region-sd/PFC+MOTOR/PFC+MOTOR_MRIcro.csv', debug=debug, box_width = 0.05, box_height = 0.07)
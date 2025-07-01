"""
File: topomap_plot.py
Author: 沈逸潇
Date: 2025-05-08
Version: 1.0
Description: 
	This script visualizes multi-channel fNIRS data using inset plots arranged
	according to normalized anatomical channel positions. It includes anatomical
	labels and supports comparison of multiple subjects.
"""

import numpy as np

def normalize_positions(pos, width, height, x_range=(0.01, 0.99), y_range=(0.05, 0.9), reverse=True):
	"""
	Normalize raw channel positions to fit within a canvas while reserving padding space 
	for inset plots, ensuring all subplots stay within bounds.

	Parameters:
		pos (ndarray): Original (n, 2) channel positions
		width (float): Width of each inset in normalized figure coordinates
		height (float): Height of each inset in normalized figure coordinates
		x_range (tuple): Target range to scale x-axis into
		y_range (tuple): Target range to scale y-axis into

	Returns:
		ndarray: Padded, normalized positions in figure coordinate space
	"""
	# Compute bounding box of original coordinates
	min_vals = pos.min(axis=0)
	max_vals = pos.max(axis=0)

	# Scale to target x/y ranges
	normalized_x = (pos[:, 0] - min_vals[0]) / (max_vals[0] - min_vals[0]) * (x_range[1] - x_range[0]) + x_range[0]
	normalized_y = (pos[:, 1] - min_vals[1]) / (max_vals[1] - min_vals[1]) * (y_range[1] - y_range[0]) + y_range[0]
	normalized = np.column_stack((normalized_x, normalized_y))

	# Add inset padding to ensure no clipping occurs
	padding = np.array([width / 2, height / 2]) * 2
	safe_pos = normalized * (1 - 2 * padding) + padding

	# Clip to [padding, 1 - padding] to stay within canvas
	safe_pos = np.clip(safe_pos, padding, 1 - padding)
	if reverse:
		safe_pos = reverse_xy(safe_pos)
	return safe_pos

def reverse_xy(pos):
	"""
	Flip x and y coordinates (mirror around center)
	in a (n, 2) array.
	"""
	x = pos[:, 0]
	y = pos[:, 1]

	x_flipped = x.max() + x.min() - x
	y_flipped = y.max() + y.min() - y

	return np.column_stack((x_flipped, y_flipped))

def get_xlim_range(*, center_sec=None, length_sec=None, start_sec=None, end_sec=None):
	"""
	Compute the time window (xlim) for a plot given one of three combinations:
	- (center_sec + length_sec)
	- (start_sec + end_sec)
	- (start_sec + length_sec)

	Returns:
		tuple: (start_sec, end_sec), center_sec

	Raises:
		ValueError: If provided argument combinations are invalid
	"""
	if center_sec is not None and length_sec is not None:
		start_sec = center_sec - length_sec / 2
		end_sec = center_sec + length_sec / 2

	elif start_sec is not None and end_sec is not None:
		center_sec = (start_sec + end_sec) / 2

	elif start_sec is not None and length_sec is not None:
		end_sec = start_sec + length_sec
		center_sec = (start_sec + end_sec) / 2

	else:
		raise ValueError("Must provide either (center_sec & length_sec), (start_sec & end_sec), or (start_sec & length_sec)")

	return (start_sec, end_sec), center_sec

def plot_labels(plt, label_template):
	"""
	Render anatomical text labels (e.g., 'Left', 'Right', 'Nose') on the figure.

	Parameters:
		plt: Matplotlib pyplot module
		label_template (list): List of dicts containing label attributes
	"""
	for label in label_template:
		text = label['text']
		x, y = label['position']
		rotation = label.get('rotation', 0)
		fontsize = label.get('fontsize', 12)
		color = label.get('color', 'black')

		plt.text(x, y, text, fontsize=fontsize, color=color, ha='center', va='center',
		         rotation=rotation, transform=plt.gcf().transFigure)


def plot_anatomical_labels(plt, template_idx=0):
	"""
	Select and plot anatomical labels based on pre-defined layout templates.

	Parameters:
		plt: Matplotlib pyplot module
		template_idx (int): Which label layout to use
	"""
	if template_idx == 0:
		label_template = [
			{'text': 'Right', 'position': (0.02, 0.5), 'rotation': 90},
			{'text': 'Left', 'position': (0.98, 0.5), 'rotation': -90},
			{'text': 'Nose', 'position': (0.5, 0.02)}
		]
	elif template_idx == 1:
		label_template = [
			{'text': 'Left', 'position': (0.02, 0.5), 'rotation': 90},
			{'text': 'Right', 'position': (0.98, 0.5), 'rotation': -90},
			{'text': 'Nose', 'position': (0.5, 0.9)}
		]
	elif template_idx == 2:
		label_template = [
			{'text': 'Right', 'position': (0.02, 0.5), 'rotation': 90},
			{'text': 'Left', 'position': (0.9, 0.5), 'rotation': -90},
			{'text': 'Nose', 'position': (0.5, 0.02)}
		]
	plot_labels(plt, label_template)


def create_inset_plots(fig, channel_pos_l, inset_width, inset_height, time, hbo_data_l, subject_name_l, xlim_range):
	"""
	Create inset line plots for each channel position in the figure.

	Parameters:
		fig (matplotlib.figure.Figure): Main figure
		channel_pos_l (ndarray): List of channel (x, y) positions
		inset_width (float): Width of each inset plot (in normalized figure coords)
		inset_height (float): Height of each inset plot
		time (ndarray): Time array
		hbo_data_l (list of ndarray): List of (channels x timepoints) HbO data arrays
		subject_name_l (list): Subject names for legend
		xlim_range (tuple): Time window to display in x-axis
	"""
	for ch_idx, (x, y) in enumerate(channel_pos_l):
		x0 = x - inset_width / 2
		y0 = y - inset_height / 2
		ax_inset = fig.add_axes([x0, y0, inset_width, inset_height])
		ax_inset.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')

		lines = []	# Store Line2D objects for each subject
		for i, hbo_data in enumerate(hbo_data_l):
			line, = ax_inset.plot(time, hbo_data[ch_idx])
			lines.append(line)

		ax_inset.set_xlim(xlim_range)
		ax_inset.set_ylim(-1, 1)
		ax_inset.tick_params(axis='both', labelsize=6, direction='in')
		ax_inset.set_title(f'Ch{ch_idx+1}', fontsize=7, pad=2)

		# Add legend only to selected channels to avoid visual clutter
		if ch_idx + 1 in {48, 36, 35, 16, 15}:
			ax_inset.legend(
				lines,
				subject_name_l,
				fontsize=5,
				loc='center left',
				bbox_to_anchor=(1.01, 0.5),
				frameon=False
			)
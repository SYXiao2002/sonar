"""
File Name: file_helper.py
Author: Yixiao Shen
Date: 2025-05-16
Purpose: define file helper
"""

import os
import shutil

import pandas as pd


def clear_folder(folder_path):
	# Check if the folder exists
	if not os.path.exists(folder_path):
		print(f"Folder '{folder_path}' does not exist.")
		return
	
	# Loop through all files and folders inside
	for filename in os.listdir(folder_path):
		file_path = os.path.join(folder_path, filename)
		try:
			# If it is a file, remove it
			if os.path.isfile(file_path) or os.path.islink(file_path):
				os.remove(file_path)
			# If it is a directory, remove it recursively
			elif os.path.isdir(file_path):
				shutil.rmtree(file_path)
		except Exception as e:
			print(f'Failed to delete {file_path}. Reason: {e}')



def save_head_csv(
	input_path: str,
	output_path: str,
	start_row: int = 0,
	end_row: int = 5000
) -> None:
	"""
	Read specified rows (from start_row to end_row) from input CSV file,
	while always keeping the first header row, then save to output CSV file.

	Args:
		input_path (str): Path to input CSV file.
		output_path (str): Path to output CSV file.
		start_row (int): Starting data row index (0-based, excluding header), default 0.
		end_row (int): Ending data row index (exclusive), default 5000.
	"""

	# read header first
	with open(input_path, 'r') as f:
		header = f.readline().strip()

	# calculate how many rows to read
	nrows = end_row - start_row

	# read specified chunk (skip start_row rows after header)
	df = pd.read_csv(input_path, skiprows=range(1, start_row + 1), nrows=nrows)

	# write to output with header
	df.to_csv(output_path, index=False, header=header.split(','))
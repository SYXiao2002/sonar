"""
File Name: file_helper.py
Author: Yixiao Shen
Date: 2025-05-16
Purpose: define file helper
"""
import os
import shutil


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
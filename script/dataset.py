"""
File Name: dataset.py
Author: Yixiao Shen
Date: 2025-05-19
Purpose: 
"""

import os
import time

from sonar.core.dataset_loader import DatasetInfo, DatasetLoader
from sonar.preprocess.mne_converter import process_dataset
from sonar.preprocess.sv_marker import read_annotations

def get_dataset(ds_dir, load_cache):
	hbo_dir = os.path.join(ds_dir, 'hbo')
	marker_file = os.path.join(ds_dir, 'marker', 'marker.csv')

	hbo_file_l = [
		os.path.join(hbo_dir, f) for f in os.listdir(hbo_dir) if f.endswith('.csv')
	]

	dataset_template =[
		DatasetInfo(f, os.path.basename(f).split('.')[0]) for f in hbo_file_l
	]

	start_time = time.time()  # Start timing
	dataset = DatasetLoader.from_csv_list(dataset_template, load_cache=load_cache)
	end_time = time.time()  # End timing
	print(f"[INFO] Dataset loaded in {end_time - start_time:.3f} seconds")
	annotations = read_annotations(marker_file)
	return dataset, annotations

def process_trainingcamp(dir, debug):
	filter_param_list=[
		(0.007, 0.04, 0.001, 0.03),
		# (0.007, 0.1, 0.001, 0.03),
		# (0.02, 0.09, 0.01, 0.1),
		# (0.007, 0.1, 0.001, 1.0),
		# (0.007, 0.1, 0.001, 2.0),
	]
	process_dataset(dir, time_shifting=2222.354 - 3, first_trigger=9, last_trigger=19, filter_param_list=filter_param_list, debug=debug, override=True, thr=30)

if __name__ == "__main__":
	process_trainingcamp('res/trainingcamp-pure', debug=False)
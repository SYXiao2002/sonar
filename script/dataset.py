"""
File Name: dataset.py
Author: Yixiao Shen
Date: 2025-05-19
Purpose: 
"""

import time

from sonar.core.dataset_loader import DatasetInfo, DatasetLoader
from sonar.preprocess.sv_marker import read_annotations

def _get_dataset(dataset_template, load_cache):
	start_time = time.time()  # Start timing
	dataset = DatasetLoader.from_csv_list(dataset_template, load_cache=load_cache)
	end_time = time.time()  # End timing
	print(f"[INFO] Dataset loaded in {end_time - start_time:.3f} seconds")
	return dataset

def get_wh_dataset(load_cache):

	dataset_template = [
		DatasetInfo('res/20250510wh_test/wjy_converted.csv', 'WJY'),
		DatasetInfo('res/20250510wh_test/wh_converted.csv', 'WH'),
	]
	annotation_path = 'res/20250510wh_test/marker.csv'
	return _get_dataset(dataset_template=dataset_template, load_cache=load_cache), read_annotations(annotation_path)

def get_trainingcamp_dataset(load_cache):

	dataset_template = [
		DatasetInfo('res/traningcamp-nirspark/HC1-hbo_converted.csv', 'HC1'),
		DatasetInfo('res/traningcamp-nirspark/HC3-hbo_converted.csv', 'HC3'),
		DatasetInfo('res/traningcamp-nirspark/HC5-hbo_converted.csv', 'HC5'),
		DatasetInfo('res/traningcamp-nirspark/HC7-hbo_converted.csv', 'HC7'),
		DatasetInfo('res/traningcamp-nirspark/HC9-hbo_converted.csv', 'HC9'),
	]
	annotation_path = 'res/traningcamp-nirspark/sv-marker/song_list.csv'
	return _get_dataset(dataset_template=dataset_template, load_cache=load_cache), read_annotations(annotation_path)


def get_trainingcamp_dataset(load_cache):

	dataset_template = [
		DatasetInfo('res/test/test-debug.csv', 'TEST1'),
		DatasetInfo('res/test/test-debug.csv', 'TEST2'),
	]
	annotation_path = 'res/test/song_list.csv'
	return _get_dataset(dataset_template=dataset_template, load_cache=load_cache), read_annotations(annotation_path)
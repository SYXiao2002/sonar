"""
File Name: dataset.py
Author: Yixiao Shen
Date: 2025-05-19
Purpose: 
"""
from sonar.preprocess.mne_converter import process_dataset

def process_trainingcamp(dir, debug):
	filter_param_list=[
		# fl, fh, ftl, fth
		# (0.02, 0.09, 0.01, 0.1),			# april
		(0.007, 0.04, 0.001, 0.03),       # may
		# (0.007, 0.1, 0.001, 0.03),
		# (0.007, 0.1, 0.001, 1.0),
		# (0.007, 0.1, 0.001, 2.0),
	]
	process_dataset(dir, time_shifting=2222.354 - 3, first_trigger=9, last_trigger=19, filter_param_list=filter_param_list, debug=debug, override=True, thr=30)

if __name__ == "__main__":
	process_trainingcamp('res/trainingcamp-mne-may', debug=False)
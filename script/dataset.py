"""
File Name: dataset.py
Author: Yixiao Shen
Date: 2025-05-19
Purpose: 
"""
from sonar.preprocess.mat_converter import mat_converter
from sonar.preprocess.mne_converter import process_dataset
from sonar.preprocess.nirspark_converter import nirspark2csv

def process_trainingcamp(ds_dir, debug, time_shifting, filter_param_list):
	process_dataset(ds_dir, time_shifting=time_shifting, first_trigger=9, last_trigger=19, filter_param_list=filter_param_list, debug=debug, override=True, thr=30)

if __name__ == "__main__":

	# nirspark
	ds_dir = 'res/trainingcamp-nirspark'
	sr = 11
	time_shifting = 2222.354
	# nirspark2csv(ds_dir, sr, time_shifting)



	# mne
	debug_filter_param_list=[
		# fl, fh, ftl, fth
		# (0.02, 0.09, 0.01, 0.1),			# april
		(0.007, 0.04, 0.001, 0.03),       # may
		# (0.01, 0.09, 0.001, 0.1),       # June
		# (0.007, 0.1, 0.001, 0.03),
		# (0.007, 0.1, 0.001, 1.0),
		# (0.007, 0.1, 0.001, 2.0),
	]
	# process_trainingcamp(ds_dir='res/trainingcamp-mne-may',filter_param_list=[(0.007, 0.04, 0.001, 0.03)],  debug=False, time_shifting=time_shifting)
	# process_trainingcamp(ds_dir='res/trainingcamp-mne-april',filter_param_list=[(0.02, 0.09, 0.01, 0.1)],  debug=False, time_shifting=time_shifting)
	process_trainingcamp(ds_dir='res/trainingcamp-mne-no-filter',filter_param_list=None,  debug=False, time_shifting=time_shifting)



	# homer3
	sub_label_l =[
		None,
		None,
		None,
		None,
		None,
		'HC3',
		'HC5',
		'HC1',
		'HC9',
		'HC7',
	]
	# mat_converter('res/trainingcamp-homer3/homerdata.mat', time_shifting=time_shifting, sub_label_l=sub_label_l)
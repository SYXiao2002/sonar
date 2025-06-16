"""
File Name: dataset.py
Author: Yixiao Shen
Date: 2025-05-19
Purpose: 
"""
from sonar.core.region_selector import RegionSelector
from sonar.preprocess.mat_converter import mat_converter
from sonar.preprocess.mne_converter import process_dataset
from sonar.preprocess.nirspark_converter import nirspark2csv

def process_trainingcamp(ds_dir, debug, time_shifting, filter_param_list, first_trigger, last_trigger, z_score=True):
	process_dataset(ds_dir, time_shifting=time_shifting, first_trigger=first_trigger, last_trigger=last_trigger, filter_param_list=filter_param_list, debug=debug, override=True, thr=30, z_score=z_score)

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
	# process_trainingcamp(ds_dir='res/trainingcamp-mne-may',filter_param_list=[(0.007, 0.04, 0.001, 0.03)],  debug=False, time_shifting=time_shifting, first_trigger=9, last_trigger=19)
	# process_trainingcamp(ds_dir='res/trainingcamp-mne-april',filter_param_list=[(0.02, 0.09, 0.01, 0.1)],  debug=False, time_shifting=time_shifting, first_trigger=9, last_trigger=19, z_score=True)
	# process_trainingcamp(ds_dir='res/trainingcamp-mne-no-filter',filter_param_list=None,  debug=False, time_shifting=time_shifting, first_trigger=9, last_trigger=19)
	# process_trainingcamp(ds_dir='res/yuanqu-mne-no-filter',filter_param_list=None,  debug=False, time_shifting=0, first_trigger=9, last_trigger=9)
	# process_trainingcamp(ds_dir='res/yuanqu-mne',filter_param_list=[(0.02, 0.09, 0.01, 0.1)],  debug=False, time_shifting=0, first_trigger=9, last_trigger=9)



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
	crop_dict = {}
	crop_dict['HC1']=RegionSelector(start_sec=2055.455, end_sec=4817.909)
	crop_dict['HC3']=RegionSelector(start_sec=2052.727, end_sec=4815.000)
	crop_dict['HC5']=RegionSelector(start_sec=2061.727, end_sec=4824.091)
	crop_dict['HC7']=RegionSelector(start_sec=95.182, end_sec=2857.727)
	crop_dict['HC9']=RegionSelector(start_sec=117.273, end_sec=2879.727)
	mat_converter('res/trainingcamp-homer3/homerdata.mat', time_shifting=0, sub_label_l=sub_label_l, crop_dict=crop_dict)
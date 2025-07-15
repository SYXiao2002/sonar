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
	# process_trainingcamp(ds_dir='res/trainingcamp-mne-luke',filter_param_list=[(0.05, 0.7, 0.02, 0.2)],  debug=False, time_shifting=time_shifting, first_trigger=9, last_trigger=19, z_score=True)
	# process_trainingcamp(ds_dir='res/tapping-luke',filter_param_list=[(0.05, 0.7, 0.02, 0.2)],  debug=False, time_shifting=0, first_trigger=None, last_trigger=None, z_score=True)
	# process_trainingcamp(ds_dir='res/tapping-luke-april',filter_param_list=[(0.02, 0.09, 0.01, 0.1)],  debug=False, time_shifting=0, first_trigger=None, last_trigger=None, z_score=True)
	# process_trainingcamp(ds_dir='res/trainingcamp-mne-no-filter',filter_param_list=None,  debug=False, time_shifting=time_shifting, first_trigger=9, last_trigger=19)
	# process_trainingcamp(ds_dir='res/yuanqu-mne-no-filter',filter_param_list=None,  debug=False, time_shifting=0, first_trigger=9, last_trigger=9)
	# process_trainingcamp(ds_dir='res/yuanqu-mne-cortivision',filter_param_list=[(0.02, 0.09, 0.01, 0.1)],  debug=False, time_shifting=0, first_trigger=9, last_trigger=9)
	# process_trainingcamp(ds_dir='res/yuanqu-mne',filter_param_list=[(0.02, 0.09, 0.01, 0.1)],  debug=False, time_shifting=0, first_trigger=9, last_trigger=9)
	process_trainingcamp(ds_dir='res/tapping0623',filter_param_list=[(0.02, 0.09, 0.01, 0.1)],  debug=False, time_shifting=0, first_trigger=1, last_trigger=None)



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
	new_names = [
		'S1_D1 hbo',
'S1_D6 hbo',
'S2_D1 hbo',
'S2_D2 hbo',
'S2_D7 hbo',
'S3_D2 hbo',
'S3_D3 hbo',
'S3_D8 hbo',
'S4_D3 hbo',
'S4_D4 hbo',
'S4_D9 hbo',
'S5_D4 hbo',
'S5_D5 hbo',
'S5_D10 hbo',
'S6_D5 hbo',
'S6_D11 hbo',
'S7_D1 hbo',
'S7_D6 hbo',
'S7_D7 hbo',
'S7_D12 hbo',
'S8_D2 hbo',
'S8_D7 hbo',
'S8_D8 hbo',
'S8_D13 hbo',
'S9_D3 hbo',
'S9_D8 hbo',
'S9_D9 hbo',
'S9_D14 hbo',
'S10_D4 hbo',
'S10_D9 hbo',
'S10_D10 hbo',
'S10_D15 hbo',
'S11_D5 hbo',
'S11_D10 hbo',
'S11_D11 hbo',
'S11_D16 hbo',
'S12_D7 hbo',
'S12_D12 hbo',
'S12_D13 hbo',
'S13_D8 hbo',
'S13_D13 hbo',
'S13_D14 hbo',
'S14_D9 hbo',
'S14_D14 hbo',
'S14_D15 hbo',
'S15_D10 hbo',
'S15_D15 hbo',
'S15_D16 hbo',
'time'
	]
	mat_converter('res/trainingcamp-homer3/homerdata.mat', time_shifting=time_shifting, sub_label_l=sub_label_l, crop_dict=crop_dict, new_names=new_names)
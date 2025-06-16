import os
from itertools import compress

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.preprocessing.nirs import optical_density

from mne_nirs.preprocessing import peak_power, scalp_coupling_index_windowed
from mne_nirs.visualisation import plot_timechannel_quality_metric

from sonar.preprocess.mne_converter import read_snirf

def main(ds_dir = 'trainingcamp-mne-april'):

	snirf_dir = os.path.join('res', ds_dir, 'snirf')
	snirf_file_l = [
		os.path.join(snirf_dir, f) for f in os.listdir(snirf_dir) if f.endswith('.snirf')
	]
	out_dir = os.path.join('out', ds_dir, 'singal_quaility')


	for snirf_file in snirf_file_l:
		sub_label = os.path.splitext(os.path.basename(snirf_file))[0]
		sub_folder = os.path.join(out_dir, sub_label)
		os.makedirs(sub_folder, exist_ok=True)

		raw_intensity = read_snirf(snirf_file)
		raw_intensity.load_data().resample(4.0, npad="auto")
		raw_od: mne.io.Raw = optical_density(raw_intensity)
		raw_od.plot(n_channels=48, show_scrollbars=False, clipping=None)
		plt.savefig(os.path.join(sub_folder, 'raw_od.png'))

		sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)
		fig, ax = plt.subplots()
		ax.hist(sci)
		ax.set(xlabel="Scalp Coupling Index", ylabel="Count", xlim=[0, 1])
		plt.savefig(os.path.join(sub_folder, 'sci.png'))

		raw_od.info["bads"] = list(compress(raw_od.ch_names, sci < 0.7))
		print(raw_od.info["bads"])

		raw_od.plot(n_channels=48, show_scrollbars=False, clipping=None)
		plt.savefig(os.path.join(sub_folder, 'raw_od_bad.png'))
		

		raw_od.plot_sensors()
		plt.savefig(os.path.join(sub_folder, 'sensors.png'))

		sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od.copy().crop(10))
		fig, ax = plt.subplots()
		ax.hist(sci)
		ax.set(xlabel="Scalp Coupling Index", ylabel="Count", xlim=[0, 1])
		plt.savefig(os.path.join(sub_folder, 'sci_crop_10s.png'))

		_, scores, times = scalp_coupling_index_windowed(raw_od, time_window=60)
		plot_timechannel_quality_metric(
			raw_od,
			scores,
			times,
			threshold=0.7,
			title="Scalp Coupling Index " "Quality Evaluation"
		)
		plt.savefig(os.path.join(sub_folder, 'sci_window_60s.png'))

		raw_od, scores, times = peak_power(raw_od, time_window=10)
		plot_timechannel_quality_metric(
			raw_od, scores, times, threshold=0.1, title="Peak Power Quality Evaluation"
		)
		plt.savefig(os.path.join(sub_folder, 'peak_power_10s.png'))

if __name__ == '__main__':
	main()
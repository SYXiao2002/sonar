import os
from typing import Sequence
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import spectrogram
from dataclasses import dataclass

from tqdm import tqdm

from sonar.core.dataset_loader import get_dataset

@dataclass
class SpectrogramResult:
	f: np.ndarray
	t: np.ndarray
	Sxx_avg: np.ndarray
	Sxx_dB: np.ndarray

class SpectrogramProcessor:
	def __init__(self, dataset, fs, out_dir, nperseg=None, noverlap=None, nfft=4096, use_cache=True, cache_path='cache/spectrogram_cache.pkl'):
		"""
		dataset: list of subjects, each subject is a list of channels (1D signals)
		"""
		self.dataset = dataset
		self.fs = fs
		self.n_channels = len(dataset[0])
		self.nperseg = nperseg or int(fs * 10)
		# self.noverlap = noverlap or int(self.nperseg * 0.95)
		self.noverlap = noverlap or int(self.nperseg ) - 1
		self.nfft = nfft
		self.out_dir = out_dir

		self.results_raw: dict[int, SpectrogramResult] = {}
		self.results_collapsed: dict[int, Sequence[float]] = {}


		if use_cache and os.path.exists(cache_path):
			self._load_cache(cache_path)
		else:
			self._compute()
			self._save_cache(cache_path)
			self._save()

	def _compute(self):
		for sub_idx, sub_label in tqdm(enumerate(self.dataset.label_l), total=len(self.dataset.label_l)):
			"""Compute and store spectrogram for a specific subject"""
			subject_data = self.dataset[sub_idx]
			Sxx_total = None

			for ch in tqdm(range(self.n_channels), desc=f"Computing spectrogram for {sub_label}", leave=False):
				signal = np.array(subject_data[ch])
				f, t, Sxx = spectrogram(
					signal,
					fs=self.fs,
					nperseg=self.nperseg,
					noverlap=self.noverlap,
					nfft=self.nfft
				)

				if Sxx_total is None:
					Sxx_total = Sxx
				else:
					Sxx_total += Sxx
				
			Sxx_avg = Sxx_total / self.n_channels
			Sxx_dB = 10 * np.log10(Sxx_avg + 1e-10)

			self.results_raw[sub_idx] = SpectrogramResult(f=f, t=t, Sxx_avg=Sxx_avg, Sxx_dB=Sxx_dB)

		self.collapse_frequency_band()

		for sub_idx, res in self.results_raw.items():
			res.t += self.dataset['time'][0]

	def collapse_frequency_band(self, f_range=(0.8, 2.0), dB_threshold=None):
		"""Collapse masked spectrogram to get mean frequency (average of max and min freq) per time point for each subject,
		only within frequency range f_lim"""
		for sub_idx, res in self.results_raw.items():
			f = res.f
			t = res.t
			Sxx_dB = res.Sxx_dB

			# Restrict frequency range
			f_mask = (f >= f_range[0]) & (f <= f_range[1])
			f_focus = f[f_mask]
			Sxx_focus = Sxx_dB[f_mask, :]

			# Apply dB threshold mask if needed
			if dB_threshold is not None:
				Sxx_focus = np.where(Sxx_focus >= dB_threshold, Sxx_focus, np.nan)

			peak_freqs = []

			for t_idx in range(Sxx_focus.shape[1]):
				col = Sxx_focus[:, t_idx]
				if np.all(np.isnan(col)):
					peak_freqs.append(np.nan)
				else:
					max_idx = np.nanargmax(col)
					peak_freqs.append(f_focus[max_idx])

			self.results_collapsed[sub_idx] = peak_freqs
			assert len(t) == len(peak_freqs)

	def _save_cache(self, path):
		"""Save computed results to disk"""
		cache_data = {
			'results_raw': self.results_raw,
			'results_collapsed': self.results_collapsed,
		}
		joblib.dump(cache_data, path)

	def _load_cache(self, path):
		"""Load results from disk"""
		cache_data = joblib.load(path)
		self.results_raw = cache_data['results_raw']
		self.results_collapsed = cache_data['results_collapsed']

	def _save(self):
		csv_dir = os.path.join('res', self.out_dir, 'spectrogram')
		os.makedirs(csv_dir, exist_ok=True)
		for sub_idx, sub_label in enumerate(self.dataset.label_l):
			mean_freq = self.results_collapsed[sub_idx]
			df = pd.DataFrame({'time': self.results_raw[sub_idx].t, 'freq': mean_freq})
			df.to_csv(os.path.join(csv_dir, f'{sub_label}.csv'), index=False)

	def plot(self, subject_idx: int, dB_threshold=-20, f_lim=(0.8, 1.8)):
		"""Plot spectrogram for a subject"""
		if subject_idx not in self.results_raw:
			raise ValueError(f"Subject {subject_idx} not computed or loaded.")
		res = self.results_raw[subject_idx]
		Sxx_dB_masked = np.where(res.Sxx_dB > dB_threshold, res.Sxx_dB, np.nan)

		plt.figure(figsize=(12, 5))
		plt.pcolormesh(res.t, res.f, Sxx_dB_masked, cmap='plasma')
		plt.ylabel('Frequency [Hz]')
		plt.xlabel('Time [s]')
		plt.plot(res.t, self.results_collapsed[subject_idx], marker='.', linestyle='-', color='b')
		plt.ylim(*f_lim)
		plt.title(f'Subject {subject_idx} Averaged Spectrogram - Power > {dB_threshold} dB')
		plt.colorbar(label='Power [dB]')
		plt.tight_layout()
		plt.show()

def run(ds_dir):
	dataset, _ = get_dataset(os.path.join('res', ds_dir), load_cache=True)
	processor = SpectrogramProcessor(dataset, fs=11, use_cache=True, cache_path=os.path.join('cache', f'{ds_dir}.pkl'), out_dir=ds_dir)

	processor.plot(subject_idx=0, dB_threshold=-20, f_lim=(1.0, 2.0))

if __name__ == '__main__':
	run(ds_dir='trainingcamp-mne-no-filter')
	# run(ds_dir='yuanqu-mne-no-filter')
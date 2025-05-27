import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

class SpectrogramPlotter:
	def __init__(self, fs=1.0, window='hann', nperseg=128, noverlap=100, nfft=256,
	             detrend='constant', scaling='density', mode='psd'):
		self.fs = fs
		self.window = window
		self.nperseg = nperseg
		self.noverlap = noverlap
		self.nfft = nfft
		self.detrend = detrend
		self.scaling = scaling
		self.mode = mode

	@staticmethod
	def generate_test_signal(fs=11, duration=100.0, freq=100.0, noise_level=0.5):
		# Generate synthetic signal: sine wave + Gaussian noise
		t = np.linspace(0, duration, int(fs * duration))
		x = np.sin(2 * np.pi * freq * t) + noise_level * np.random.randn(t.size)
		return t, x

	def compute(self, x):
		f, t, Sxx = spectrogram(
			x,
			fs=self.fs,
			window=self.window,
			nperseg=self.nperseg,
			noverlap=self.noverlap,
			nfft=self.nfft,
			detrend=self.detrend,
			scaling=self.scaling,
			mode=self.mode
		)
		return f, t, Sxx

	def plot(self, x, dB=True):
		f, t_spec, Sxx = self.compute(x)
		if dB:
			Sxx = 10 * np.log10(Sxx + 1e-10)

		plt.pcolormesh(t_spec, f, Sxx, shading='gouraud')
		plt.ylabel('Frequency [Hz]')
		plt.xlabel('Time [sec]')
		plt.title('Spectrogram')
		plt.colorbar(label='Power/Frequency (dB/Hz)' if dB else 'Power/Frequency')
		plt.tight_layout()
		plt.show()

if __name__ == "__main__":
	fs = 11
	duration = 100.0
	freq = 100.0
	noise_level = 0.5
	t, x = SpectrogramPlotter.generate_test_signal(fs, duration, freq, noise_level)
	sp = SpectrogramPlotter(fs, window='hann', nperseg=128, noverlap=100, nfft=256,)
	sp.plot(x)
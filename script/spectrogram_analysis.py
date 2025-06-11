
# column name: time,SPO2,Pulse,HR
ppg_csv_path = 'res/yuanqu-mne/heartrate_PPG/Sub2.csv'

# column name: time,ch_count


# column name: time,freq
spectrogram_csv_path = 'res/yuanqu-mne-no-filter/spectrogram/Sub2.csv'


from matplotlib import pyplot as plt
import pandas as pd

# plot
plt.figure(figsize=(10, 4))

# get data
ppg_df = pd.read_csv(ppg_csv_path)
intensity_df = pd.read_csv(intensity_csv_path)
spectrogram_df = pd.read_csv(spectrogram_csv_path)

# get HR, ch_count, freq and their time
ppg_time = ppg_df['time'].values
ppg_hr = ppg_df['HR'].values

intensity_time = intensity_df['time'].values
intensity_value = intensity_df['ch_count'].values

spectrogram_time = spectrogram_df['time'].values
spectrogram_value = spectrogram_df['freq'].values

# no-szore before plot
# 创建3个子图
fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
fig.suptitle('Heatrate from PPG vs Heartrate from fNIRS')

# Subplot: PPG HR
axs[0].plot(ppg_time, ppg_hr, label='HR')
axs[0].set_ylabel('Raw HR\nfrom PPG (bpm)')

# Subplot: fNIRS Spectrogram HR
axs[1].plot(spectrogram_time, spectrogram_value, label='value')
axs[1].set_ylabel('Raw HR\nfrom fNIRS (Hz)')


# Subplot: fNIRS Intensity
axs[2].plot(intensity_time, intensity_value, label='intensity')
axs[2].set_ylabel('Raw MTS Intensity\nfrom fNIRS')
axs[2].set_xlabel('Time (s)')

for ax in axs:
	ax.grid(True)
	# ax.set_ylim(-2, 2)

# 可选：紧凑布局
plt.tight_layout()
# plt.show()
plt.savefig('out/no-zscore.png', dpi=600)
plt.close()


# z-score before plot
ppg_hr = (ppg_hr - ppg_hr.mean()) / ppg_hr.std()
spectrogram_value = (spectrogram_value - spectrogram_value.mean()) / spectrogram_value.std()
# intensity_value = (intensity_value - intensity_value.mean()) / intensity_value.std()

# 创建3个子图
fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
fig.suptitle('Heatrate from PPG vs Heartrate from fNIRS')

# Subplot: PPG HR
axs[0].plot(ppg_time, ppg_hr, label='HR')
axs[0].set_ylabel('z-scored HR\nfrom PPG')

# Subplot: fNIRS Spectrogram HR
axs[1].plot(spectrogram_time, spectrogram_value, label='value')
axs[1].set_ylabel('z-scored HR\nfrom fNIRS')

# Subplot: fNIRS Intensity
axs[2].plot(intensity_time, intensity_value, label='intensity')
axs[2].set_ylabel('MTS Intensity\nfrom fNIRS')
axs[2].set_xlabel('Time (s)')

for ax in axs:
	ax.grid(True)

axs[0].set_ylim(-2, 2)
axs[1].set_ylim(-2, 2)

# 可选：紧凑布局
plt.tight_layout()
# plt.show()
plt.savefig('out/z-scored.png', dpi=600)
plt.close()

import os
import numpy as np
import mne

import random

from sonar.preprocess.snirf_metadata import get_metadata_dict

# ==== 1. 加载文件 ====
ref_path = 'res/brain-region-sd/temp/3D_plot/RAW_REF.csv'
head_path = 'res/brain-region-sd/temp/3D_plot/RAW_HEAD.csv'
name2meta_ref, _ = get_metadata_dict(ref_path)
name2meta_head, _ = get_metadata_dict(head_path)

# ==== 2. 提取 fiducials ====
raw_fids = {}
for name, meta in name2meta_ref.items():
	pos = np.array(meta.pos)
	if name.lower().startswith('nz'):
		raw_fids['nasion'] = pos
	elif name.lower().startswith('al'):
		raw_fids['lpa'] = pos
	elif name.lower().startswith('ar'):
		raw_fids['rpa'] = pos

# ==== 3. 判断坐标单位 ====
dist = np.linalg.norm(raw_fids['lpa'] - raw_fids['rpa'])

# 判断坐标单位并选择缩放比例
if 0.1 <= dist <= 0.2:
	scale = 1.0		# 单位为米
elif 10 <= dist <= 20:
	scale = 0.01	# 单位为厘米
elif 100 <= dist <= 200:
	scale = 0.001	# 单位为毫米
else:
	raise ValueError(f"Unusual LPA–RPA distance: {dist:.3f}. Cannot determine unit scale.")

print(f"Distance between LPA and RPA: {dist:.3f}")
print(f"Assumed unit scale: {scale}")

# ==== 4. 应用缩放到 fiducials ====
fiducials_dict = {k: v * scale for k, v in raw_fids.items()}

# ==== 5. 缩放通道坐标 ====
ch_pos = {name: np.array(meta.pos) * scale for name, meta in name2meta_head.items()}

# ==== 6. 构建 montage ====
montage = mne.channels.make_dig_montage(
	ch_pos=ch_pos,
	nasion=fiducials_dict['nasion'],
	lpa=fiducials_dict['lpa'],
	rpa=fiducials_dict['rpa'],
	coord_frame='head'
)

# ==== 4. 构造 Raw 对象并加载 montage ====
info = mne.create_info(ch_names=list(ch_pos.keys()), sfreq=1.0, ch_types='eeg')
raw = mne.io.RawArray(np.zeros((len(ch_pos), 1)), info)
raw.set_montage(montage)

# ==== 5. Coregistration ====
subjects_dir = 'res/brain-region-sd/parcellation'
os.makedirs(subjects_dir, exist_ok=True)

mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir, verbose=True)
mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir, accept=True)

coreg = mne.coreg.Coregistration(
	raw.info, subject='fsaverage', subjects_dir=subjects_dir, fiducials='estimated'
)
coreg.fit_fiducials(lpa_weight=1.0, nasion_weight=1.0, rpa_weight=1.0)

# ==== 6. 可视化脑图并绘制通道 ====
brain = mne.viz.Brain(
	subject='fsaverage',
	subjects_dir=subjects_dir,
	background='w',
	cortex='0.5'
)

# ==== 7. 添加通道点（传感器）====
brain.add_sensors(
	info=raw.info,			# 包含 montage 的 Raw.info
	trans=coreg.trans,		# 使用配准矩阵
)


# ==== 7. 加载脑区标签 ====
labels = mne.read_labels_from_annot('fsaverage', parc='aparc', hemi='both', subjects_dir=subjects_dir)

def generate_random_rgba(n):
	colors = []
	for _ in range(n):
		r = random.random()
		g = random.random()
		b = random.random()
		a = 1
		colors.append((r, g, b, a))
	return colors

colors = generate_random_rgba(len(labels))
for label, color in zip(labels, colors):
	brain.add_label(label, color=color, borders=False)

# ==== 8. 展示结果 ====
brain.show_view(azimuth=180, elevation=80, distance=550)
input("Press Enter to continue...")
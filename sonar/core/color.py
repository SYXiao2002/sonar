import hashlib
import seaborn as sns

color_pool = sns.color_palette("tab20")  # "Set2", "Paired", "husl", "dark", "colorblind"
num_colors = len(color_pool)

def get_color_from_label(label):
	if type(label) != str:
		label = str(label)
	# Use md5 for a stable hash
	hash_val = int(hashlib.md5(label.encode()).hexdigest(), 16)
	return color_pool[hash_val % num_colors]
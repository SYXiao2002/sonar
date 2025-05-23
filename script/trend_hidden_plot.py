import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('out/total_curve_increasing_6.4s.csv')

# Filter the DataFrame to only include sub_label HC1, HC3, and HC5
df = df[df['sub_label'].isin(['HC1', 'HC3', 'HC5'])]

plt.figure(figsize=(10, 6))

for label, group in df.groupby('sub_label'):
	# Sort values by center_time to ensure a proper line
	group = group.sort_values('center_time')
	
	# Min-max normalization for center_time and count_sum
	center_time_min = group['center_time'].min()
	center_time_max = group['center_time'].max()
	group['center_time_norm'] = (group['center_time'] - center_time_min) / (center_time_max - center_time_min)

	count_sum_min = group['count_sum'].min()
	count_sum_max = group['count_sum'].max()
	group['count_sum_norm'] = (group['count_sum'] - count_sum_min) / (count_sum_max - count_sum_min)

	# Plot normalized values
	plt.plot(group['center_time_norm'], group['count_sum_norm'], label=label)

# Set axis labels and title
plt.xlabel('Normalized Center Time')  # x-axis label
plt.ylabel('Normalized Count Sum')   # y-axis label
plt.title('Normalized Count Sum over Normalized Center Time by Sub Label')
plt.legend()
plt.tight_layout()
plt.show()

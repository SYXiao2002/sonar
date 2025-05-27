import h5py

def check_or_create_landmark_labels(snirf_file):
	"""
	Check if the /nirs/probe/landmarkLabels dataset exists in the SNIRF file.
	If it does not exist, create an empty dataset.

	:param snirf_file: str, the path to the SNIRF file
	"""
	# Open the SNIRF file in read/write mode
	with h5py.File(snirf_file, 'r+') as f:
		# Check if the /nirs/probe/landmarkLabels dataset exists
		if '/nirs/probe/landmarkLabels' in f:
			# If it exists, read and decode the content
			labels = f['/nirs/probe/landmarkLabels'][:]
			decoded_labels = [label.decode('utf-8') for label in labels]  # Decode byte strings to UTF-8
			print("Existing landmark labels:", decoded_labels)
		else:
			# If it does not exist, create an empty dataset
			print("/nirs/probe/landmarkLabels dataset not found, creating an empty list")
			f.create_dataset('/nirs/probe/landmarkLabels', data=[], dtype='S10')  # 'S10' for string data type
			print("Empty landmarkLabels dataset created")

	# Re-open the file in read mode to verify the changes
	with h5py.File(snirf_file, 'r') as f:
		# Verify if the dataset now exists
		if '/nirs/probe/landmarkLabels' in f:
			labels = f['/nirs/probe/landmarkLabels'][:]
			decoded_labels = [label.decode('utf-8') for label in labels]
			print("Verified landmark labels:", decoded_labels)
		else:
			print("Verification failed, dataset not found")

# Example usage
if __name__ == "__main__":
	# Replace with the actual SNIRF file path
	snirf_file = 'res/snirf/May-SeaLife-Kong-Chapter01.snirf'
	check_or_create_landmark_labels(snirf_file)

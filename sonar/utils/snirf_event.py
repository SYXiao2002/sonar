import csv
import os
import random
from typing import Any, List, Tuple
import mne
import matplotlib.pyplot as plt

from sonar.core.trigger_region import TriggerRegion
from sonar.preprocess.fix_snirf import check_or_create_landmark_labels

def get_trigger_envelope(trigger_times, min_region_duration, max_trigger_interval, trigger_values=None)-> Tuple[Any, List[TriggerRegion]]:
	"""
	Plot the trigger envelope showing start and end times of regions of rising edges.
	
	:param trigger_times: List of trigger timestamps in seconds.
	:param trigger_values: List of trigger values corresponding to the trigger_times.
	:param min_region_duration: Minimum duration for a region to be plotted (in seconds).
	:param max_trigger_interval: Maximum time interval allowed between adjacent triggers to group them into one region.
	"""
	regions_raw_l = []  # List to store all regions of rising edges
	regions_valid_l = []  # List to store valid regions
	current_region = []  # Temporary list to build regions
	value_to_color = {}  # Dictionary to map trigger_value to a specific color

	if trigger_values is None:
		trigger_values = ['NAN'] * len(trigger_times)

	# Group triggers into regions based on the max_trigger_interval and matching trigger_values
	for t, v in zip(trigger_times, trigger_values):
		if len(current_region) == 0:
			current_region.append((t, v))  # Add the first trigger
		elif t - current_region[-1][0] < max_trigger_interval and v == current_region[-1][1]:
			current_region.append((t, v))  # Add to the current region if within time interval and same trigger value
		else:
			# Add the completed region to the list and start a new region
			regions_raw_l.append((current_region[0][0], current_region[-1][0], current_region[0][1]))  # Add start and end time, and trigger value
			current_region = [(t, v)]  # Start a new region with the current trigger

	# Add the last region if it exists
	if len(current_region) > 0:
		regions_raw_l.append((current_region[0][0], current_region[-1][0], current_region[0][1]))

	# Plot the envelope for each region
	plt.figure(figsize=(24, 6))
	text_position = 0.2  # Initialize text position for the first region
	
	for i, (start, end, value) in enumerate(regions_raw_l, start=1):
		if end - start < min_region_duration:
			continue  # Skip regions that are too short

		# Assign a color to the trigger_value if it's not assigned yet
		if value not in value_to_color:
			# Generate a random dark color for each unique trigger_value
			value_to_color[value] = (random.uniform(0, 0.5), random.uniform(0, 0.5), random.uniform(0, 0.5))
		
		random_color = value_to_color[value]  # Use the assigned color for this value
		
		# Plot the region with the generated color
		plt.axvspan(start, end, color=random_color, alpha=0.6, label=f"Region {i}")
		
		# Add text annotations for the start and end times of each region
		mid_point = (start + end) / 2  # Calculate the midpoint for text positioning

		region = TriggerRegion(start=start, end=end, trigger_value=value)
		regions_valid_l.append(region)
		text = region.to_text(spliter="\n")

		plt.text(mid_point, text_position, 
				 text, 
				 ha='center', va='center', color=random_color, fontsize=10, 
				 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

		text_position += 0.2  # Update the text position for the next region
		if text_position > 0.8:
			text_position = 0.2  # Reset text position if it exceeds the limit

	# Customize the plot appearance
	plt.xlabel("Time (s)")
	plt.ylabel("Regions")
	plt.grid()
	plt.xlim(0, max(trigger_times) + 10)  # Extend x-axis to show some margin after the last trigger
	return regions_valid_l


def read_snirf_file(filepath):
	"""
	Reads a SNIRF file and returns the raw data.
	Args:
		filepath (str): Path to the SNIRF file.
	Returns:
		mne.io.Raw: Raw data object.
	"""
	check_or_create_landmark_labels(filepath)
	raw = mne.io.read_raw_snirf(filepath)
	time_as_index = raw.time_as_index
	return raw

def summarize_events(raw):
	"""
	Generates and prints a summary report for events in the raw data.
	Args:
		raw (mne.io.Raw): Raw data object containing annotations.
	"""
	try:
		events, event_id = mne.events_from_annotations(raw)
		event_count = len(events)

		print(f"\nTotal Events: {event_count}\n")

		if event_count > 0:
			print("First 5 Events:")
			for i, event in enumerate(events[:5]):
				print(f"  Event {i + 1}:")
				print(f"    Time: {event[0]} samples")
				print(f"    Event ID: {event[2]}")

			if event_count > 5:
				print("\nLast 5 Events:")
				for i, event in enumerate(events[-5:]):
					print(f"  Event {event_count - 5 + i + 1}:")
					print(f"    Time: {event[0]} samples")
					print(f"    Event ID: {event[2]}")

			print("\nEvent ID Mapping:")
			for key, value in event_id.items():
				print(f"  {key}: {value}")
		else:
			print("No events found in the file.")
	except ValueError:
		print("\nNo events found in annotations.")


def plot(raw, sr, snirf_file):
    # Extract events and their corresponding dictionary from raw data annotations
    events, event_dict = mne.events_from_annotations(raw)
    
    # Reverse the event dictionary for easy lookup
    event_dict_reversed = {v: k for k, v in event_dict.items()}
    
    # Calculate trigger times and values
    trigger_times = [float(e[0] / sr) for e in events]
    trigger_values = [event_dict_reversed.get(e[2]) for e in events]
    
    # Generate the trigger envelope
    regions_json_l = get_trigger_envelope(trigger_times=trigger_times, trigger_values=trigger_values, min_region_duration=10, max_trigger_interval=3)
    
    # Set plot title
    plt.title(f"Trigger Envelope for '{snirf_file}'")
    
    # Prepare file paths for saving the plot and JSON output
    filename = os.path.splitext(os.path.basename(snirf_file))[0]
    export_folder = os.path.dirname(snirf_file)
    
    fig_path = os.path.join(export_folder, f"{filename}_trigger_envelope.png")
    json_path = os.path.join(export_folder, f"{filename}_trigger_envelope.json")
    txt_path = os.path.join(export_folder, f"trigger_envelope_all.txt")

    # Save plot and export trigger region data as JSON
    plt.savefig(fig_path)
    TriggerRegion.export_to_json(regions_json_l, json_path)
    TriggerRegion.export_to_txt(subject_idx=filename, trigger_regions=regions_json_l, filepath=txt_path)


def export_events_to_csv(raw, snirf_file):
    """
    Exports events from the raw data to a CSV file.
    Args:
        raw (mne.io.Raw): Raw data object containing annotations.
        snirf_file (str): Path to the SNIRF file.
    """
    events, event_id = mne.events_from_annotations(raw)
    event_count = len(events)
    
    # Create the output file path
    filename = os.path.basename(snirf_file)
    output_path = os.path.join("out", f"{filename}_events.csv")
    
    # Open the CSV file for writing
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header
        writer.writerow(['Event Number', 'Time (samples)', 'Event ID'])
        
        # Write event data
        for i, event in enumerate(events):
            writer.writerow([i + 1, event[0], event[2]])
    
    print(f"Events exported to {output_path}")
		


def process_snirf(snirf_file):
	raw = read_snirf_file(snirf_file)
	# summarize_events(raw)
	sr = raw.info['sfreq']
	plot(raw, sr, snirf_file=snirf_file)
	# export_events_to_csv(raw, snirf_file)
	
def main():
	dir = 'res/trainingcamp-mne-april-audience/snirf'

	snirf_l=[]
	for file in os.listdir(dir):    
		if file.endswith(".snirf"):
			snirf_l.append(os.path.join(dir, file))

	for snirf_file in snirf_l:
		process_snirf(snirf_file)

if __name__ == "__main__":
	main()

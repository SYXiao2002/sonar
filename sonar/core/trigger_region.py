import json
import os
from typing import List, Union

import pandas as pd

class TriggerRegion:
	def __init__(self, start, end, trigger_value):
		"""Initialize a trigger region with start time, end time, and trigger value."""
		self.start = start
		self.end = end
		self.trigger_value = trigger_value

	def to_text(self, spliter="\n"):
		"""Return a formatted text representation of the trigger region."""
		return f"Start: {self.start:.3f}s{spliter}End: {self.end:.3f}s{spliter}Duration: {self.end - self.start:.3f}s{spliter}Trigger Value: {self.trigger_value}"

	def to_json(self):
		"""Return a dictionary representation of the trigger region."""
		return {
			"Start": f"{self.start:.3f}",
			"End": f"{self.end:.3f}",
			"Duration": f"{self.end - self.start:.3f}",
			"Trigger Value": str(self.trigger_value)
		}
	
	def to_pandas(self, subject_idx=None) -> pd.DataFrame:
		"""
		Convert TriggerRegion to a Pandas DataFrame.
		Returns a DataFrame with column format.
		"""
		return pd.DataFrame([{
			"Subject Index": subject_idx,
			"Start (s)": self.start,
			"End (s)": self.end,
			"Duration (s)": self.end - self.start,
			"Trigger Value": self.trigger_value
		}])

	def __str__(self):
		"""Make the object printable, defaulting to text representation."""
		return self.to_text()

	@staticmethod
	def export_to_json(trigger_regions: Union[List['TriggerRegion'], 'TriggerRegion'], filepath: str):
		"""
		Export a list of TriggerRegion objects (or a single TriggerRegion) to a JSON file.

		:param trigger_regions: List of TriggerRegion objects or a single TriggerRegion object.
		:param filepath: Output JSON file path.
		"""
		# Ensure trigger_regions is a list
		if isinstance(trigger_regions, TriggerRegion):
			trigger_regions = [trigger_regions]
		
		# Convert each TriggerRegion to JSON format
		data = [region.to_json() for region in trigger_regions]
		
		# Write data to JSON file
		with open(filepath, "w", encoding="utf-8") as f:
			json.dump(data, f, indent=4)

	@staticmethod
	def export_to_txt(trigger_regions: Union[List['TriggerRegion'], 'TriggerRegion'], filepath: str, subject_idx=None):
		"""
		Export a list of TriggerRegion objects (or a single TriggerRegion) to a TXT file in column format.

		- If the file **does not exist**, write data with headers.
		- If the file **exists**, append data **without headers**.

		:param trigger_regions: List of TriggerRegion objects or a single TriggerRegion object.
		:param filepath: Output TXT file path.
		:param subject_idx: Optional subject index for better tracking.
		"""
		# Ensure trigger_regions is a list
		if isinstance(trigger_regions, TriggerRegion):
			trigger_regions = [trigger_regions]

		# Convert to pandas DataFrame
		data = pd.concat([region.to_pandas(subject_idx) for region in trigger_regions], ignore_index=True)

		# Check if file exists
		file_exists = os.path.exists(filepath)

		# Append to file, write header only if file does not exist
		with open(filepath, "a", encoding="utf-8") as f:
			data.to_csv(f, index=False, sep="\t", header=not file_exists, float_format="%.3f")

if __name__ == "__main__":
	region = TriggerRegion(1.23, 4.56, 5)
	region_l=[region, region]

	TriggerRegion.export_to_json(region_l, "trigger_regions.json")
	TriggerRegion.export_to_txt(region_l, "trigger_regions.txt", "dd")
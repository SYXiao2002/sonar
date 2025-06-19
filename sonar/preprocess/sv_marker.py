from typing import NamedTuple, Sequence
import csv

import pandas as pd

class Annotation(NamedTuple):
	start: float
	value: int
	duration: float
	label: str

def read_annotations(csv_path: str) -> Sequence[Annotation]:
	annotation_l: Sequence[Annotation] = []


	df = pd.read_csv(csv_path)

	if 'TIME' in df.columns and 'DURATION' in df.columns:
		for _, row in df.iterrows():
			start = float(row["TIME"])
			duration = float(row["DURATION"])
			label = row["LABEL"]
			value = int(row["VALUE"])
			annotation_l.append(Annotation(start=start, value=value, duration=duration, label=label))
	elif 'start_sec' in df.columns and 'length_sec' in df.columns:
		for _, row in df.iterrows():
			start = float(row["start_sec"])
			duration = float(row["length_sec"])
			label = 'None'
			value = 0
			annotation_l.append(Annotation(start=start, value=value, duration=duration, label=label))
	else:
		Warning("No valid columns found in csv file")
	return annotation_l
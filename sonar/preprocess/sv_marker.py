from typing import List, NamedTuple
import csv

class Annotation(NamedTuple):
	start: float
	value: int
	duration: float
	label: str

def read_annotations(path: str) -> List[Annotation]:
	annotations = []
	with open(path, 'r') as f:
		reader = csv.DictReader(f)
		for row in reader:
			start = float(row["TIME"])
			duration = float(row["DURATION"])
			label = row["LABEL"]
			value = int(row["VALUE"])
			annotations.append(Annotation(start=start, value=value, duration=duration, label=label))
	return annotations

"""
File Name: manual_labeling.py
Author: Yixiao Shen
Date: 2025-07-15
Purpose: For "Hui Chuang" devices only, create 2D points by channel index, and thus create snirf_metadata.csv
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import json

class FNIRSLabelingTool:
	def __init__(self, image_path):
		self.image_path = image_path
		self.img = mpimg.imread(image_path)
		self.clicked_points = []
		self.point_artists = []
		self.preview_lines = []
		self.enter_pressed = False

		self._init_plot()
		self._connect_events()

	def _init_plot(self):
		self.fig, self.ax = plt.subplots(figsize=(8, 8))
		self.ax.imshow(self.img)
		self.ax.set_title("Move mouse to preview, left-click to select. Press Enter when done.")

	def _connect_events(self):
		self.fig.canvas.mpl_connect('motion_notify_event', self._update_preview)
		self.fig.canvas.mpl_connect('button_press_event', self._onclick)
		self.fig.canvas.mpl_connect('key_press_event', self._on_key)

	def _update_preview(self, event):
		if not event.inaxes:
			return
		for line in self.preview_lines:
			line.remove()
		self.preview_lines.clear()
		x, y = event.xdata, event.ydata
		h_line = self.ax.axhline(y, color='gray', linestyle='--', linewidth=0.8)
		v_line = self.ax.axvline(x, color='gray', linestyle='--', linewidth=0.8)
		self.preview_lines.extend([h_line, v_line])
		plt.draw()

	def _onclick(self, event):
		if event.button == 1 and event.inaxes:
			x, y = event.xdata, event.ydata
			self.clicked_points.append((x, y))
			for line in self.preview_lines:
				line.remove()
			self.preview_lines.clear()
			cross_artist, = self.ax.plot(x, y, marker='+', color='red', markersize=10, markeredgewidth=2)
			text_artist = self.ax.text(x + 5, y + 5, str(len(self.clicked_points)), color='blue', fontsize=12, weight='bold')
			self.point_artists.append((cross_artist, text_artist))
			plt.draw()

	def _on_key(self, event):
		if event.key == 'enter':
			self.enter_pressed = True
		elif event.key == 'backspace':
			self.undo_last_point()

	def undo_last_point(self):
		if self.clicked_points:
			self.clicked_points.pop()
			cross_artist, text_artist = self.point_artists.pop()
			cross_artist.remove()
			text_artist.remove()
			plt.draw()

	def run(self):
		print("Move mouse to preview cross, left-click to select a point.")
		print("Press Enter when done, or press Backspace to undo the last point.")

		while not self.enter_pressed:
			plt.pause(0.1)

		self._save_results()

	def _save_results(self):
		labels = []
		for i, (x, y) in enumerate(self.clicked_points):
			label = str(i + 1)
			labels.append((label, int(x), int(y)))
			print(f"Labeled: {label} -> ({int(x)}, {int(y)})")

		coord_dict = {label: {"x": x, "y": y} for label, x, y in labels}
		with open("fnirs_coordinates.json", "w") as f:
			json.dump(coord_dict, f, indent=4)
		print("Saved to fnirs_coordinates.json")

		df = pd.DataFrame(labels, columns=["Label", "X", "Y"])
		df.to_csv("fnirs_coordinates.csv", index=False)
		print("Saved to fnirs_coordinates.csv")

		self.ax.set_title("Labeling complete. Close this window manually.")
		plt.savefig('fnirs_labeling_result.png', dpi=600)
		plt.close()


if __name__ == "__main__":
	tool = FNIRSLabelingTool('res/brain-region-sd/temp/sd.png')  # replace with your image path
	# 注意输出，是以左上角为origin
	tool.run()

import os
import cv2
import numpy as np
from scipy.io import loadmat, savemat
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

class AnnotationEditor:
    def __init__(self, image_path, annotation_path):
        self.image_path = image_path
        self.annotation_path = annotation_path
        self.points = []
        self.history = []  # For undo/redo
        self.redo_stack = []  # Stack for redo actions

        # Load the image
        self.img = cv2.imread(image_path)
        if self.img is None:
            raise ValueError(f"Unable to read image: {image_path}")
        self.img_height, self.img_width = self.img.shape[:2]  # Store original dimensions
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # Load existing annotation points
        self.load_annotations()

        # Initialize Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.imshow(self.img, extent=[0, self.img_width, self.img_height, 0])  # Match image's pixel coordinates
        self.ax.set_title("Manual Annotation Tool")
        self.ax.axis("off")

        # Plot existing points
        self.scatter = self.ax.scatter(
            self.points[:, 0], self.points[:, 1], c='red', s=20, label='Annotated Points'
        )

        # Connect event listeners
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        # Add buttons for actions
        self.add_buttons()

        # Show the figure
        plt.legend()
        plt.show()

    def load_annotations(self):
        """
        Load annotation points from the .mat file.
        """
        annotation_data = loadmat(self.annotation_path)
        points = annotation_data.get('points', None)
        if points is None:
            raise ValueError(f"No points found in annotation file: {self.annotation_path}")
        self.points = np.array(points)

    def save_annotations(self):
        """
        Save the updated points back to the .mat file.
        """
        savemat(self.annotation_path, {'points': self.points})
        print(f"Annotations saved to {self.annotation_path}")

    def add_buttons(self):
        """
        Add action buttons for point, remove, undo, redo, and save.
        """
        ax_undo = plt.axes([0.7, 0.01, 0.1, 0.05])
        ax_redo = plt.axes([0.81, 0.01, 0.1, 0.05])
        ax_save = plt.axes([0.59, 0.01, 0.1, 0.05])

        self.btn_undo = Button(ax_undo, 'Undo')
        self.btn_undo.on_clicked(self.undo)

        self.btn_redo = Button(ax_redo, 'Redo')
        self.btn_redo.on_clicked(self.redo)

        self.btn_save = Button(ax_save, 'Save')
        self.btn_save.on_clicked(self.save)

    def on_click(self, event):
        """
        Handle mouse click events for adding or removing points.
        """
        if event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata
        if event.button == 1:  # Left click to add a point
            self.add_point(x, y)
        elif event.button == 3:  # Right click to remove a point
            self.remove_point(x, y)

    def add_point(self, x, y):
        """
        Add a new point to the annotations.
        """
        self.history.append(self.points.copy())
        self.redo_stack.clear()
        self.points = np.vstack([self.points, [x, y]])
        self.update_plot()

    def remove_point(self, x, y):
        """
        Remove the nearest point to the clicked location.
        """
        if len(self.points) == 0:
            return

        # Find the closest point to (x, y)
        distances = np.linalg.norm(self.points - np.array([x, y]), axis=1)
        nearest_index = np.argmin(distances)

        self.history.append(self.points.copy())
        self.redo_stack.clear()
        self.points = np.delete(self.points, nearest_index, axis=0)
        self.update_plot()

    def undo(self, event=None):
        """
        Undo the last action.
        """
        if self.history:
            self.redo_stack.append(self.points.copy())
            self.points = self.history.pop()
            self.update_plot()
        else:
            print("Nothing to undo.")

    def redo(self, event=None):
        """
        Redo the last undone action.
        """
        if self.redo_stack:
            self.history.append(self.points.copy())
            self.points = self.redo_stack.pop()
            self.update_plot()
        else:
            print("Nothing to redo.")

    def save(self, event=None):
        """
        Save the current annotations to the file.
        """
        self.save_annotations()

    def update_plot(self):
        """
        Update the scatter plot with the current points.
        """
        self.scatter.set_offsets(self.points)
        self.fig.canvas.draw_idle()


def select_image_and_annotation():
    """
    Open a file dialog to select an image and its corresponding annotation file.
    """
    Tk().withdraw()  # Hide the root window
    image_path = askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not image_path:
        print("No image selected. Exiting.")
        return None, None

    image_dir, image_name = os.path.split(image_path)
    image_base, _ = os.path.splitext(image_name)
    annotation_dir = os.path.join(os.path.dirname(image_dir), 'annotations')
    annotation_path = os.path.join(annotation_dir, f"{image_base}.mat")

    if not os.path.exists(annotation_path):
        print(f"Annotation file not found for image: {image_path}")
        return None, None

    return image_path, annotation_path


def main():
    image_path, annotation_path = select_image_and_annotation()
    if image_path and annotation_path:
        AnnotationEditor(image_path, annotation_path)


if __name__ == "__main__":
    main()
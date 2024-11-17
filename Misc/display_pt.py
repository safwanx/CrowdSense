import os
import cv2
import numpy as np
from scipy.io import loadmat
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from matplotlib import pyplot as plt

def select_image():
    """
    Open a file dialog to select an image file.
    """
    Tk().withdraw()  # Hide the root window
    file_path = askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    return file_path

def get_annotation_path(image_path):
    """
    Get the corresponding annotation file for the selected image.
    Assumes annotations are in a sibling 'annotations' folder with the same base name.
    """
    image_dir, image_name = os.path.split(image_path)
    image_base, _ = os.path.splitext(image_name)
    annotation_dir = os.path.join(os.path.dirname(image_dir), 'annotations')
    annotation_path = os.path.join(annotation_dir, f"{image_base}.mat")
    
    if not os.path.exists(annotation_path):
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
    return annotation_path

def display_points_on_image(image_path, annotation_path):
    """
    Load an image and its corresponding points from an annotation file, 
    and display the points on the image as it is (without resizing or cropping).
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to read image: {image_path}")
    
    # Convert to RGB for plotting
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width = img_rgb.shape[:2]
    
    # Load the annotation file
    annotation_data = loadmat(annotation_path)
    points = annotation_data.get('points', None)
    
    if points is None:
        raise ValueError(f"No points found in annotation file: {annotation_path}")
    
    # Filter points to ensure they're within the image bounds (optional, for visualization clarity)
    valid_points = [(x, y) for x, y in points if 0 <= x < img_width and 0 <= y < img_height]
    valid_points = np.array(valid_points)
    
    # Display the image with points
    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    if valid_points.size > 0:
        plt.scatter(valid_points[:, 0], valid_points[:, 1], c='red', s=20, label='Annotated Points')
    else:
        print("Warning: No valid points within image bounds.")
    plt.title("Image with Annotations")
    plt.axis("off")
    plt.legend()
    plt.show()

def main():
    try:
        print("Select an image file:")
        image_path = select_image()
        if not image_path:
            print("No image file selected. Exiting.")
            return
        
        print(f"Selected image: {image_path}")
        
        # Get the annotation file path
        annotation_path = get_annotation_path(image_path)
        print(f"Corresponding annotation file: {annotation_path}")
        
        # Display the image with points
        display_points_on_image(image_path, annotation_path)
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
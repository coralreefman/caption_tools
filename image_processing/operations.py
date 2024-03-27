from PIL import Image
import numpy as np
from skimage import filters
from skimage.measure import label
import cv2
import matplotlib.pyplot as plt

def resize(image, max_size=2048):

    width, height = image.size

    # Check if resizing is needed
    if width > max_size or height > max_size:
        # Calculate the aspect ratio
        aspect_ratio = float(width) / float(height)

        # Determine the new width and height while maintaining the aspect ratio
        if width > height:
            new_width = max_size
            new_height = int(max_size / aspect_ratio)
        else:
            new_height = max_size
            new_width = int(max_size * aspect_ratio)

        # Resize the image and return the resized version
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        return resized_image
    else:
        return image
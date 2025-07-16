# src/visualization.py
import numpy as np
from PIL import Image, ImageDraw

def create_overlay(image, mask, color, alpha=0.5):
    """
    Creates a transparent overlay of a mask on an image.

    Args:
        image (PIL.Image): The base image.
        mask (numpy.ndarray): The binary mask (0s and 1s).
        color (tuple): The RGB color for the overlay.
        alpha (float): The transparency of the overlay.

    Returns:
        PIL.Image: The image with the overlay.
    """
    # Create a new image for the overlay, with the specified color
    overlay = Image.new('RGBA', image.size, color + (0,))
    draw = ImageDraw.Draw(overlay)

    # Convert mask to a boolean array to draw the overlay
    mask_bool = mask.astype(bool)

    # Create a color mask
    color_mask = np.zeros((*mask.shape, 4), dtype=np.uint8)
    color_mask[mask_bool] = color + (int(255 * alpha),)

    # Convert the numpy array to an Image
    color_mask_img = Image.fromarray(color_mask, 'RGBA')

    # Composite the overlay onto the original image
    return Image.alpha_composite(image.convert('RGBA'), color_mask_img)
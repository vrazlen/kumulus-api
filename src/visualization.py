# src/visualization.py
import numpy as np
from PIL import Image, ImageDraw
# In src/visualization.py
import torch
from torchvision.transforms.functional import to_pil_image

def save_tensor_as_image(tensor: torch.Tensor, file_path: str):
    """
    Saves a 3-channel PyTorch tensor as a standard image file (e.g., PNG).
    Handles normalization and data type conversion. [cite: 55]

    Args:
        tensor (torch.Tensor): The input tensor, expected shape (C, H, W).
        file_path (str): The path to save the output image.
    """
    # Ensure tensor is on the CPU for processing with PIL [cite: 56]
    if tensor.is_cuda:
        tensor = tensor.cpu()

    # If the tensor is normalized, scale it to 0-255 and convert to byte type [cite: 57]
    if tensor.dtype == torch.float32 or tensor.dtype == torch.float64:
        tensor = torch.clamp(tensor, 0, 1)
        tensor = tensor.mul(255).byte()

    # Convert the tensor to a PIL Image and save in a lossless format [cite: 58, 59]
    pil_image = to_pil_image(tensor)
    pil_image.save(file_path, format='PNG')
    print(f"Successfully saved high-fidelity image to {file_path}")

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
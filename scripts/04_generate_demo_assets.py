# scripts/04_generate_demo_assets.py
import os
import torch
from PIL import Image
import numpy as np
import segmentation_models_pytorch as smp
from torchvision import transforms as T

# Import our visualization function
from src.visualization import create_overlay

# --- Configuration ---
MODEL_PATH = 'models/unet_slum_segmentation.pth'
IMAGE_DIR = 'data/processed/patches' # Corrected to look in 'patches'
ASSET_DIR = 'app/assets'

# Choose a few tiles to use for the demo.
# These filenames should exist in your data/processed/patches folder.
DEMO_TILES = {
    "District A": "patch_256_512.tif",
    "District B": "patch_1024_0.tif",
    "District C": "patch_512_1024.tif"
}

# --- Main Logic ---
def generate_assets():
    """
    Loads the trained model, runs inference on demo tiles, and saves
    the output images for the Streamlit dashboard.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the trained model
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None, 
        in_channels=3,
        classes=1,
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    print("Model loaded successfully.")
    os.makedirs(ASSET_DIR, exist_ok=True)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for district_name, tile_filename in DEMO_TILES.items():
        print(f"Processing {district_name}...")
        image_path = os.path.join(IMAGE_DIR, tile_filename)

        if not os.path.exists(image_path):
            print(f"Warning: Demo tile {tile_filename} not found in {IMAGE_DIR}. Skipping.")
            continue

        # Load the base image and convert to RGB
        base_image = Image.open(image_path).convert("RGB")
        
        # Prepare image for model
        image_tensor = transform(base_image).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            output = model(image_tensor)
            preds = torch.sigmoid(output)
            mask_array = (preds > 0.5).cpu().numpy().squeeze()

        # --- CORRECT INDENTATION ---
        # Define the overlay color (R, G, B) and create the overlay
        overlay_color = (255, 0, 0) 
        overlay_image = create_overlay(base_image, mask_array, color=overlay_color)

        # Save the base and overlay images to the assets folder
        base_output_path = os.path.join(ASSET_DIR, f"{district_name.lower().replace(' ', '_')}_base.png")
        overlay_output_path = os.path.join(ASSET_DIR, f"{district_name.lower().replace(' ', '_')}_overlay.png")

        base_image.save(base_output_path)
        overlay_image.save(overlay_output_path)

        print(f"Saved assets for {district_name} to {ASSET_DIR}")

if __name__ == '__main__':
    generate_assets()
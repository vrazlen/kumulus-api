# scripts/03_create_patches.py
import os
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from PIL import Image
import numpy as np

# --- Configuration ---
CLIPPED_RGB_TIF = 'data/processed/jakarta_clipped_rgb.tif'
GROUND_TRUTH_SHP = 'data/raw/jakarta_slum_boundaries.shp'

IMAGE_PATCH_DIR = 'data/processed/patches'
MASK_PATCH_DIR = 'data/processed/masks' # The directory we were missing

PATCH_SIZE = 256

# --- Main Logic ---
def create_image_and_mask_patches():
    """
    Creates aligned patches for both satellite imagery and ground truth masks.
    """
    if not os.path.exists(CLIPPED_RGB_TIF):
        print(f"Error: Clipped RGB TIF not found at {CLIPPED_RGB_TIF}")
        return

    if not os.path.exists(GROUND_TRUTH_SHP):
        print(f"Error: Ground truth shapefile not found at {GROUND_TRUTH_SHP}")
        return

    # Create output directories
    os.makedirs(IMAGE_PATCH_DIR, exist_ok=True)
    os.makedirs(MASK_PATCH_DIR, exist_ok=True)
    print(f"Directories '{IMAGE_PATCH_DIR}' and '{MASK_PATCH_DIR}' are ready.")

    # Open the clipped RGB satellite image
    with rasterio.open(CLIPPED_RGB_TIF) as src:
        meta = src.meta.copy()

        # Load and reproject the ground truth shapefile
        gdf = gpd.read_file(GROUND_TRUTH_SHP)
        if gdf.crs != src.crs:
            print("Reprojecting ground truth vector to match raster CRS...")
            gdf = gdf.to_crs(src.crs)

        # --- Rasterize the vector data to create a full-size mask ---
        print("Rasterizing ground truth shapefile to create mask...")
        mask_geoms = [(geom, 1) for geom in gdf.geometry]

        full_mask = rasterize(
            shapes=mask_geoms,
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0, # Background value
            all_touched=True,
            dtype=rasterio.uint8
        )

        # --- Iterate and create patches for BOTH image and mask ---
        print("Creating and saving image and mask patches...")
        saved_count = 0
        for j in range(0, src.height, PATCH_SIZE):
            for i in range(0, src.width, PATCH_SIZE):
                if i + PATCH_SIZE > src.width or j + PATCH_SIZE > src.height:
                    continue

                # Define the window
                window = rasterio.windows.Window(i, j, PATCH_SIZE, PATCH_SIZE)

                # Read image patch and create PIL Image
                img_patch_data = src.read(window=window)
                img_patch_rgb = np.transpose(img_patch_data, (1, 2, 0))
                img_patch = Image.fromarray(img_patch_rgb, 'RGB')

                # Crop mask patch from the full rasterized mask
                mask_patch_data = full_mask[j:j + PATCH_SIZE, i:i + PATCH_SIZE]
                mask_patch = Image.fromarray(mask_patch_data, 'L')

                # Save both corresponding patches
                patch_filename_base = f"patch_{i}_{j}"
                img_patch.save(os.path.join(IMAGE_PATCH_DIR, f"{patch_filename_base}.tif"))
                mask_patch.save(os.path.join(MASK_PATCH_DIR, f"{patch_filename_base}_mask.tif"))
                saved_count += 1

        print(f"Successfully created and saved {saved_count} image/mask pairs.")

if __name__ == '__main__':
    create_image_and_mask_patches()
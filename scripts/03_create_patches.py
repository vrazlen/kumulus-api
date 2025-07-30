# scripts/03_create_patches.py
import os
import shutil
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from rasterio.features import rasterize
from PIL import Image
import numpy as np

print("--- Intelligent & Uniform Patch Creation Script Started ---")

# --- Configuration ---
RASTER_PATH = 'data/processed/jakarta_clipped_rgb.tif'
GROUND_TRUTH_PATH = 'data/raw/slum_polygons_jakarta.shp'
OUTPUT_PATCHES_DIR = 'data/processed/patches'
OUTPUT_MASKS_DIR = 'data/processed/masks'
PATCH_SIZE = 256

def clean_and_prepare_dirs():
    """Removes old patches and creates clean output directories."""
    for path in [OUTPUT_PATCHES_DIR, OUTPUT_MASKS_DIR]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
    print(f"Clean directories '{OUTPUT_PATCHES_DIR}' and '{OUTPUT_MASKS_DIR}' are ready.")

def rasterize_ground_truth(raster_src, gdf):
    """Rasterize vector ground truth to match the raster's grid."""
    print("Rasterizing ground truth shapefile to create mask...")
    if gdf.crs != raster_src.crs:
        gdf = gdf.to_crs(raster_src.crs)
    
    mask_raster = rasterize(
        shapes=gdf.geometry.tolist(),
        out_shape=raster_src.shape,
        transform=raster_src.transform,
        fill=0,
        default_value=255,
        dtype=rasterio.uint8
    )
    return mask_raster

def scale_to_8bit(array_16bit):
    """Scales a 16-bit NumPy array to an 8-bit array."""
    max_val = 3000
    scaled_array = np.clip(array_16bit, 0, max_val)
    scaled_array = (scaled_array / max_val) * 255.0
    return scaled_array.astype(np.uint8)

def create_and_save_patches(raster_src, mask_raster):
    """Create and save corresponding, full-sized image and mask patches."""
    print("Creating and saving image and mask patches...")
    count = 0
    for j in range(0, raster_src.height, PATCH_SIZE):
        for i in range(0, raster_src.width, PATCH_SIZE):
            window = Window(i, j, PATCH_SIZE, PATCH_SIZE)
            
            # --- CRITICAL FIX: Ensure we only process full-sized patches ---
            if window.width != PATCH_SIZE or window.height != PATCH_SIZE:
                continue

            img_patch_16bit = raster_src.read(window=window)
            
            if np.all(img_patch_16bit == 0):
                continue
            
            mask_patch_data = mask_raster[window.row_off:window.row_off+PATCH_SIZE, window.col_off:window.col_off+PATCH_SIZE]
            
            if np.sum(mask_patch_data) > 0:
                img_patch_8bit = scale_to_8bit(img_patch_16bit)
                img_patch_rgb = np.transpose(img_patch_8bit, (1, 2, 0))

                img_patch = Image.fromarray(img_patch_rgb)
                mask_patch = Image.fromarray(mask_patch_data)
                
                patch_filename = f"patch_{j}_{i}.tif"
                img_patch.save(os.path.join(OUTPUT_PATCHES_DIR, patch_filename))
                mask_patch.save(os.path.join(OUTPUT_MASKS_DIR, patch_filename))
                
                count += 1
            
    print(f"Successfully created and saved {count} uniform, relevant image/mask pairs.")

def main():
    """Main workflow to generate training patches."""
    clean_and_prepare_dirs()
    
    with rasterio.open(RASTER_PATH) as src:
        gdf = gpd.read_file(GROUND_TRUTH_PATH)
        full_mask_raster = rasterize_ground_truth(src, gdf)
        create_and_save_patches(src, full_mask_raster)

if __name__ == "__main__":
    main()
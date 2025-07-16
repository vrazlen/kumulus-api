# scripts/02_prepare_geodata.py
import os
import glob
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import box

# --- Configuration ---
SENTINEL_DIR = 'data/raw/sentinel'
AOI_FILE = 'data/raw/jakarta_aoi_correct.geojson'
PROCESSED_DIR = 'data/processed'
CLIPPED_RASTER_OUTPUT_PATH = os.path.join(PROCESSED_DIR, 'jakarta_clipped_rgb.tif')

# --- Main Logic ---
def find_band_file(path, band_id):
    """Finds a specific band file within a directory."""
    search_pattern = os.path.join(path, f'*{band_id}*.jp2')
    files = glob.glob(search_pattern)
    if not files:
        raise FileNotFoundError(f"No file found for band {band_id} in {path}")
    return files[0]

def create_rgb_and_clip():
    """
    Loads Sentinel-2 Red, Green, and Blue bands, creates an RGB composite,
    and clips it to the AOI's bounding box.
    """
    print("Locating Sentinel-2 .SAFE directory...")
    safe_folder = next((d for d in os.listdir(SENTINEL_DIR) if d.endswith('.SAFE')), None)
    if not safe_folder:
        raise FileNotFoundError(f".SAFE folder not found in {SENTINEL_DIR}")

    granule_path = os.path.join(SENTINEL_DIR, safe_folder, 'GRANULE')
    granule_folder = os.listdir(granule_path)[0]
    r10m_path = os.path.join(granule_path, granule_folder, 'IMG_DATA', 'R10m')

    print("Finding Red (B04), Green (B03), and Blue (B02) band files...")
    red_file = find_band_file(r10m_path, 'B04')
    green_file = find_band_file(r10m_path, 'B03')
    blue_file = find_band_file(r10m_path, 'B02')

    # Read metadata from one of the bands to use as a template
    with rasterio.open(red_file) as src:
        meta = src.meta.copy()

    # Update metadata for a 3-channel (RGB) GeoTIFF
    meta.update(count=3, driver='GTiff')

    print("Creating and saving temporary RGB composite...")
    temp_rgb_path = os.path.join(PROCESSED_DIR, 'temp_rgb.tif')
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    with rasterio.open(temp_rgb_path, 'w', **meta) as dst:
        with rasterio.open(red_file) as src_r:
            dst.write(src_r.read(1), 1) # Red channel
        with rasterio.open(green_file) as src_g:
            dst.write(src_g.read(1), 2) # Green channel
        with rasterio.open(blue_file) as src_b:
            dst.write(src_b.read(1), 3) # Blue channel

    print("Clipping RGB composite to AOI...")
    vector_gdf = gpd.read_file(AOI_FILE)

    with rasterio.open(temp_rgb_path) as composite_src:
        if vector_gdf.crs != composite_src.crs:
            vector_gdf = vector_gdf.to_crs(composite_src.crs)

        bounds = vector_gdf.total_bounds
        bbox_poly = box(*bounds)

        clipped_image, clipped_transform = mask(
            dataset=composite_src,
            shapes=[bbox_poly],
            crop=True
        )

        clipped_meta = composite_src.meta.copy()
        clipped_meta.update({
            "height": clipped_image.shape[1],
            "width": clipped_image.shape[2],
            "transform": clipped_transform,
        })

        print(f"Saving final clipped RGB raster to {CLIPPED_RASTER_OUTPUT_PATH}...")
        with rasterio.open(CLIPPED_RASTER_OUTPUT_PATH, "w", **clipped_meta) as dest:
            dest.write(clipped_image)

    # Clean up temporary file
    os.remove(temp_rgb_path)
    print("Data curation complete. RGB image is ready.")

if __name__ == '__main__':
    create_rgb_and_clip()
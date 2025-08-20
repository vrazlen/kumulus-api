# scripts/02_prepare_geodata.py
import geopandas as gpd
from shapely.geometry import box
import os
import rasterio
from rasterio.mask import mask
import glob

# --- Configuration ---
AOI_FILE_RAW = "data/raw/jakarta_boundary.geojson"
AOI_FILE_PREPARE = "data/processed/jakarta_aoi_prepared.geojson"
DOWNLOADED_RASTER_DIR = "data/raw/sentinel"
PROCESSED_RASTER_PATH = "data/processed/jakarta_sentinel_clipped_4band.tif"
TARGET_CRS = "EPSG:4326"

# --- Helper Functions ---
def find_band_files(product_path):
    """Finds the paths for B2, B3, B4, and B8 (10m resolution) bands."""
    search_path = os.path.join(product_path, "GRANULE", "*", "IMG_DATA", "R10m", "*_B0[2348]_10m.jp2")
    files = glob.glob(search_path)
    
    band_files = {
        "B02": next((f for f in files if "_B02_" in f), None),
        "B03": next((f for f in files if "_B03_" in f), None),
        "B04": next((f for f in files if "_B04_" in f), None),
        "B08": next((f for f in files if "_B08_" in f), None),
    }
    return band_files

# --- Main Execution ---
def main():
    """
    Prepares the AOI, stacks the core Sentinel-2 bands (B, G, R, NIR),
    and clips the final raster to the prepared AOI.
    """
    print("--- Starting Full Geodata Preparation ---")

    # --- Step 1: Prepare and Save the AOI ---
    if not os.path.exists(AOI_FILE_RAW):
        print(f"Error: Raw AOI file not found at {AOI_FILE_RAW}. Please run 00_acquire_aoi.py.")
        return

    print(f"Preparing AOI from {AOI_FILE_RAW}...")
    gdf = gpd.read_file(AOI_FILE_RAW)
    
    if gdf.crs.to_string() != TARGET_CRS:
        gdf = gdf.to_crs(TARGET_CRS)

    os.makedirs(os.path.dirname(AOI_FILE_PREPARE), exist_ok=True)
    gdf.to_file(AOI_FILE_PREPARE, driver='GeoJSON')
    print(f"Standardized AOI saved to {AOI_FILE_PREPARE}")

    # --- Step 2: Find the downloaded .SAFE product directory ---
    product_dir = next((os.path.join(DOWNLOADED_RASTER_DIR, d) for d in os.listdir(DOWNLOADED_RASTER_DIR) if d.endswith(".SAFE")), None)
    
    if not product_dir:
        print(f"Error: No .SAFE product directory found in {DOWNLOADED_RASTER_DIR}")
        return
    print(f"Found product directory: {product_dir}")

    # --- Step 3: Locate and Stack Bands ---
    band_files = find_band_files(product_dir)
    if not all(band_files.values()):
        print(f"Error: Missing required bands. Found: {band_files}")
        return
        
    with rasterio.open(band_files["B04"]) as src: meta = src.meta.copy()
    meta.update(count=4, driver='GTiff')

    temp_stacked_path = os.path.join(DOWNLOADED_RASTER_DIR, "stacked_temp.tif")
    print("Stacking bands into a temporary 4-band GeoTIFF...")
    with rasterio.open(temp_stacked_path, 'w', **meta) as dst:
        with rasterio.open(band_files["B02"]) as src_b: dst.write(src_b.read(1), 1)
        with rasterio.open(band_files["B03"]) as src_g: dst.write(src_g.read(1), 2)
        with rasterio.open(band_files["B04"]) as src_r: dst.write(src_r.read(1), 3)
        with rasterio.open(band_files["B08"]) as src_n: dst.write(src_n.read(1), 4)

    # --- Step 4: Clip the stacked raster using the PREPARED AOI ---
    print(f"Clipping the 4-band raster using the prepared AOI...")
    vector_gdf = gpd.read_file(AOI_FILE_PREPARE)
    
    with rasterio.open(temp_stacked_path) as src:
        if vector_gdf.crs != src.crs:
            vector_gdf = vector_gdf.to_crs(src.crs)

        bbox_poly = box(*vector_gdf.total_bounds)
        clipped_image, clipped_transform = mask(dataset=src, shapes=[bbox_poly], crop=True)
        clipped_meta = src.meta.copy()
        clipped_meta.update({
            "height": clipped_image.shape[1],
            "width": clipped_image.shape[2],
            "transform": clipped_transform,
        })

        print(f"Saving final clipped 4-band raster to {PROCESSED_RASTER_PATH}")
        with rasterio.open(PROCESSED_RASTER_PATH, 'w', **clipped_meta) as dest:
            dest.write(clipped_image)

    # --- Step 5: Clean up ---
    os.remove(temp_stacked_path)
    print("Cleaned up temporary files.")
    print("--- Geodata Preparation Complete ---")

if __name__ == "__main__":
    main()
# scripts/02_prepare_geodata.py
import os
import glob
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import box

# --- Configuration ---
SENTINEL_DIR = 'data/raw/sentinel'
AOI_FILE = 'data/raw/jakarta_aoi_correct.geojson'
PROCESSED_DIR = 'data/processed'
# This will be our primary 3-channel output
CLIPPED_RGB_RASTER_OUTPUT_PATH = os.path.join(PROCESSED_DIR, 'jakarta_clipped_rgb.tif')

def stack_rgb_bands(product_path: str, output_path: str):
    """
    Finds Sentinel-2 B4, B3, B2 bands in a .SAFE product folder,
    stacks them, and saves as a 3-channel GeoTIFF.
    """
    # This search pattern is robust to find the 10m resolution bands [cite: 34]
    band_search_pattern = os.path.join(product_path, 'GRANULE', '*', 'IMG_DATA', 'R10m', '*_B0*_10m.jp2')
    band_files = glob.glob(band_search_pattern)

    # Find the specific bands for Red, Green, and Blue [cite: 20]
    b4_path = next((f for f in band_files if '_B04_' in f), None) # Red
    b3_path = next((f for f in band_files if '_B03_' in f), None) # Green
    b2_path = next((f for f in band_files if '_B02_' in f), None) # Blue

    if not all([b4_path, b3_path, b2_path]):
        raise FileNotFoundError(f"Could not find all required RGB bands (B02, B03, B04) in {product_path}")
    
    print("Found band files:")
    print(f" - Red (B04): {os.path.basename(b4_path)}")
    print(f" - Green (B03): {os.path.basename(b3_path)}")
    print(f" - Blue (B02): {os.path.basename(b2_path)}")

    # Use metadata from one band as a template for the output GeoTIFF [cite: 35]
    with rasterio.open(b4_path) as src:
        meta = src.meta.copy()

    # Update metadata for a 3-channel RGB output [cite: 35]
    meta.update({
        'driver': 'GTiff',
        'count': 3,
        'dtype': 'uint16' # Sentinel-2 L2A data is typically 16-bit
    })

    print(f"Stacking bands into new GeoTIFF: {output_path}")
    with rasterio.open(output_path, 'w', **meta) as dst:
        with rasterio.open(b4_path) as src_r, \
             rasterio.open(b3_path) as src_g, \
             rasterio.open(b2_path) as src_b:
            dst.write(src_r.read(1), 1)  # Red to band 1
            dst.write(src_g.read(1), 2)  # Green to band 2
            dst.write(src_b.read(1), 3)  # Blue to band 3
    print("Successfully created 3-channel RGB GeoTIFF.")


def clip_raster_to_aoi(input_raster, output_raster, aoi_path):
    """Clips a raster to the bounding box of a vector AOI."""
    print(f"Clipping {os.path.basename(input_raster)} to AOI...")
    vector_gdf = gpd.read_file(aoi_path)
    
    with rasterio.open(input_raster) as src:
        if vector_gdf.crs != src.crs:
            vector_gdf = vector_gdf.to_crs(src.crs)

        # Get the bounding box of the AOI to use for clipping
        bbox_poly = box(*vector_gdf.total_bounds)
        
        clipped_image, clipped_transform = mask(
            dataset=src,
            shapes=[bbox_poly],
            crop=True
        )
        
        clipped_meta = src.meta.copy()
        clipped_meta.update({
            "height": clipped_image.shape[1],
            "width": clipped_image.shape[2],
            "transform": clipped_transform,
        })

        print(f"Saving final clipped RGB raster to {output_raster}...")
        with rasterio.open(output_raster, "w", **clipped_meta) as dest:
            dest.write(clipped_image)

def main():
    """Main workflow for preparing geodata."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    print("Locating Sentinel-2 .SAFE directory...")
    safe_folders = glob.glob(os.path.join(SENTINEL_DIR, '*.SAFE'))
    if not safe_folders:
        raise FileNotFoundError(f".SAFE folder not found in {SENTINEL_DIR}")
    
    # Use the first .SAFE folder found
    product_path = safe_folders[0]
    print(f"Processing product: {product_path}")
    
    temp_stacked_rgb_path = os.path.join(PROCESSED_DIR, 'temp_stacked_rgb.tif')
    
    # 1. Stack bands into a full 3-channel image
    stack_rgb_bands(product_path, temp_stacked_rgb_path)
    
    # 2. Clip the stacked image to our AOI
    clip_raster_to_aoi(temp_stacked_rgb_path, CLIPPED_RGB_RASTER_OUTPUT_PATH, AOI_FILE)

    # 3. Clean up the temporary full-sized raster
    os.remove(temp_stacked_rgb_path)
    
    print("\nData preparation complete. Clipped 3-channel RGB image is ready.")

if __name__ == '__main__':
    main()
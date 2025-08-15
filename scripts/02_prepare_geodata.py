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
# The output is now a 4-band B, G, R, NIR image
CLIPPED_RASTER_OUTPUT_PATH = os.path.join(PROCESSED_DIR, 'jakarta_clipped_4band.tif')

def find_band_file(path: str, band_id: str) -> str:
    """Robustly finds a specific band file within a Sentinel-2 product."""
    # Search in the R10m directory for B2, B3, B4, B8
    search_path = os.path.join(path, 'GRANULE', '*', 'IMG_DATA', 'R10m')
    search_pattern = os.path.join(search_path, f'*_{band_id}_10m.jp2')
    files = glob.glob(search_pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f"No file found for band {band_id} in {path}")
    return files[0]

def main():
    """Main workflow for preparing geodata."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    print("Locating Sentinel-2 .SAFE directory...")
    safe_folders = glob.glob(os.path.join(SENTINEL_DIR, '*.SAFE'))
    if not safe_folders:
        raise FileNotFoundError(f".SAFE folder not found in {SENTINEL_DIR}")
    
    product_path = safe_folders[0]
    print(f"Processing product: {os.path.basename(product_path)}")
    
    # Find all required bands
    print("Finding Blue (B02), Green (B03), Red (B04), and NIR (B08) bands...")
    blue_file = find_band_file(product_path, 'B02')
    green_file = find_band_file(product_path, 'B03')
    red_file = find_band_file(product_path, 'B04')
    nir_file = find_band_file(product_path, 'B08')

    # Create a temporary stacked 4-band file
    temp_4band_path = os.path.join(PROCESSED_DIR, 'temp_4band.tif')
    print("Stacking B, G, R, NIR bands into a temporary file...")
    with rasterio.open(red_file) as src:
        meta = src.meta.copy()
    
    meta.update(count=4, driver='GTiff')

    with rasterio.open(temp_4band_path, 'w', **meta) as dst:
        with rasterio.open(blue_file) as src_b: dst.write(src_b.read(1), 1)
        with rasterio.open(green_file) as src_g: dst.write(src_g.read(1), 2)
        with rasterio.open(red_file) as src_r: dst.write(src_r.read(1), 3)
        with rasterio.open(nir_file) as src_n: dst.write(src_n.read(1), 4)

    # Clip the stacked raster to the AOI
    print(f"Clipping raster to AOI defined in {AOI_FILE}...")
    vector_gdf = gpd.read_file(AOI_FILE)
    
    with rasterio.open(temp_4band_path) as src:
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

        print(f"Saving final clipped 4-band raster to {CLIPPED_RASTER_OUTPUT_PATH}...")
        with rasterio.open(CLIPPED_RASTER_OUTPUT_PATH, "w", **clipped_meta) as dest:
            dest.write(clipped_image)
    
    os.remove(temp_4band_path)
    print("\nData preparation complete. Clipped 4-band image is ready for analysis.")

if __name__ == '__main__':
    main()
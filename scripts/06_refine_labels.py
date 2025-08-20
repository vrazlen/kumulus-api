# scripts/06_refine_labels.py
import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape

# --- Configuration ---
INPUT_IMAGE_PATH = "data/processed/jakarta_sentinel_clipped_4band.tif"
INPUT_ANNOTATIONS_PATH = "data/processed/jakarta_annotations_clipped.gpkg"
OUTPUT_MULTICLASS_PATH = "data/processed/jakarta_annotations_multiclass.gpkg"

NDWI_THRESHOLD = 0.0
NDVI_THRESHOLD = 0.4
# NEW: Filter out tiny, noisy polygons. Area is in square meters.
MINIMUM_AREA_THRESHOLD = 1000.0 

def calculate_index(band1, band2):
    """Calculates a normalized difference index (like NDVI or NDWI)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        index = (band1 - band2) / (band1 + band2)
    return np.nan_to_num(index)

def vectorize_mask(mask, transform, crs):
    """Converts a binary raster mask into a GeoDataFrame of polygons."""
    mask_shapes = shapes(mask, mask=(mask == 1), transform=transform)
    geometries = [shape(geom) for geom, val in mask_shapes if val == 1]
    if not geometries:
        return gpd.GeoDataFrame(geometry=[], crs=crs)
    return gpd.GeoDataFrame(geometry=geometries, crs=crs)

def main():
    """
    Generates a multi-class annotation set, filtering negative classes by area
    to reduce noise and combat class imbalance.
    """
    print("--- Starting Multi-Class Label Generation (with Filtering) ---")

    # --- 1. Load Data ---
    with rasterio.open(INPUT_IMAGE_PATH) as src:
        transform = src.transform; crs = src.crs
        green_band = src.read(2).astype(float)
        red_band = src.read(3).astype(float)
        nir_band = src.read(4).astype(float)
    annotations_gdf = gpd.read_file(INPUT_ANNOTATIONS_PATH)

    # --- 2. Generate and FILTER Water and Vegetation Polygons ---
    print("Generating and filtering negative class polygons...")
    ndwi = calculate_index(green_band, nir_band)
    water_mask = (ndwi > NDWI_THRESHOLD).astype(np.uint8)
    water_gdf = vectorize_mask(water_mask, transform, crs)
    water_gdf['class'] = 'water'
    water_gdf = water_gdf[water_gdf.geometry.area > MINIMUM_AREA_THRESHOLD]

    ndvi = calculate_index(nir_band, red_band)
    veg_mask = (ndvi > NDVI_THRESHOLD).astype(np.uint8)
    veg_gdf = vectorize_mask(veg_mask, transform, crs)
    veg_gdf['class'] = 'vegetation'
    veg_gdf = veg_gdf[veg_gdf.geometry.area > MINIMUM_AREA_THRESHOLD]
    
    print(f"  -> Kept {len(water_gdf)} water and {len(veg_gdf)} vegetation polygons after area filtering.")

    # --- 3. Refine Settlement Polygons ---
    print("Refining settlement polygons...")
    negative_mask_gdf = gpd.pd.concat([water_gdf, veg_gdf], ignore_index=True)
    if not negative_mask_gdf.empty:
        negative_union = negative_mask_gdf.unary_union
        refined_settlements_gdf = annotations_gdf.overlay(
            gpd.GeoDataFrame(geometry=[negative_union], crs=crs), how='difference'
        )
    else:
        refined_settlements_gdf = annotations_gdf.copy()
        
    refined_settlements_gdf['class'] = 'informal_settlement'
    
    # --- 4. Combine and Save Final Dataset ---
    final_gdf = gpd.pd.concat(
        [refined_settlements_gdf, water_gdf, veg_gdf], ignore_index=True
    ).dropna(subset=['geometry'])

    print(f"Generated {len(final_gdf)} total polygons across 3 classes.")
    final_gdf.to_file(OUTPUT_MULTICLASS_PATH, driver="GPKG")
    print(f"--- Multi-Class Label Generation Complete ---")

if __name__ == "__main__":
    main()
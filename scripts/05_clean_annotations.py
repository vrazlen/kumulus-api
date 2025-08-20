# scripts/05_clean_annotations.py
import geopandas as gpd
import rasterio
import os
from shapely.geometry import box

# --- Configuration ---
INPUT_IMAGE_PATH = "data/processed/jakarta_sentinel_clipped_4band.tif"
INPUT_ANNOTATIONS_PATH = "data/processed/jakarta_annotations.gpkg"
OUTPUT_CLEANED_PATH = "data/processed/jakarta_annotations_clipped.gpkg"

def main():
    """
    Cleans the annotation dataset by clipping it to the spatial extent of the
    satellite image. This ensures perfect spatial consistency between the labels
    and the training data by handling CRS mismatches.
    """
    print("--- Starting Annotation Cleaning Script ---")

    # --- 1. Load the annotations and the raster image ---
    print(f"Loading annotations from: {INPUT_ANNOTATIONS_PATH}")
    annotations_gdf = gpd.read_file(INPUT_ANNOTATIONS_PATH)
    print(f"  -> Found {len(annotations_gdf)} raw polygons in CRS: {annotations_gdf.crs}")

    print(f"Reading spatial extent and CRS from: {INPUT_IMAGE_PATH}")
    with rasterio.open(INPUT_IMAGE_PATH) as src:
        image_bounds = src.bounds
        image_crs = src.crs
    print(f"  -> Image CRS is: {image_crs}")

    # --- 2. CORRECTED: Reproject annotations to match the image CRS ---
    if annotations_gdf.crs != image_crs:
        print(f"CRS mismatch detected. Reprojecting annotations from {annotations_gdf.crs} to {image_crs}...")
        annotations_gdf = annotations_gdf.to_crs(image_crs)
        print("  -> Reprojection complete.")

    # --- 3. Create a clipping mask from the raster's extent ---
    clipping_polygon = box(*image_bounds)
    mask_gdf = gpd.GeoDataFrame([1], geometry=[clipping_polygon], crs=image_crs)

    # --- 4. Perform the clipping operation ---
    print("Clipping annotations to the image's spatial extent...")
    clipped_annotations_gdf = gpd.clip(annotations_gdf, mask_gdf)
    print(f"  -> Produced {len(clipped_annotations_gdf)} cleaned polygons.")

    # --- 5. Save the cleaned dataset ---
    if clipped_annotations_gdf.empty:
        print("\nWarning: The cleaned dataset is empty. This could mean none of the raw annotations overlap with the satellite image area.")
    else:
        print(f"Saving cleaned annotations to: {OUTPUT_CLEANED_PATH}")
        clipped_annotations_gdf.to_file(OUTPUT_CLEANED_PATH, driver="GPKG")

    print("\n--- Annotation Cleaning Complete ---")

if __name__ == "__main__":
    main()
# scripts/03_generate_training_data_auto.py
import os
import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely.geometry import Polygon
from tqdm import tqdm

# --- Configuration ---
INPUT_SETTLEMENT_LIST = "data/raw/settlement_list.txt"
OUTPUT_ANNOTATION_FILE = "data/processed/jakarta_annotations.gpkg"
TARGET_CRS = "EPSG:4326" # WGS 84 a standard for web mapping

# --- Main Execution ---
def main():
    """
    Automates the creation of a training dataset by fetching administrative
    boundaries for a list of informal settlements.
    """
    print("--- Starting Automated Training Data Generation ---")

    if not os.path.exists(INPUT_SETTLEMENT_LIST):
        print(f"Error: Input file not found at {INPUT_SETTLEMENT_LIST}")
        return

    # --- 1. Parse the list of settlement names ---
    print(f"Reading settlement list from {INPUT_SETTLEMENT_LIST}...")
    with open(INPUT_SETTLEMENT_LIST, 'r') as f:
        locations = [line.strip() for line in f if line.strip()]
    print(f"Found {len(locations)} locations to process.")

    # --- 2. Geocode boundaries for each location ---
    all_polygons = []
    failed_locations = []

    print("Fetching boundaries from OpenStreetMap via osmnx...")
    # Use tqdm for a progress bar as this will take time.
    for location_query in tqdm(locations, desc="Geocoding Locations"):
        try:
            # geocode_to_gdf returns a GeoDataFrame with the boundary
            gdf = ox.geocode_to_gdf(location_query)
            
            # Ensure it's a polygon and add to our list
            if not gdf.empty and isinstance(gdf.iloc[0].geometry, Polygon):
                # We only need the geometry
                all_polygons.append(gdf.iloc[0].geometry)
            else:
                failed_locations.append(location_query)
        except Exception:
            # This handles cases where OSMN/Nominatim can't find the location
            failed_locations.append(location_query)
            
    print(f"\nSuccessfully fetched {len(all_polygons)} boundaries.")
    if failed_locations:
        print(f"Could not find boundaries for {len(failed_locations)} locations:")
        for loc in failed_locations:
            print(f"  - {loc}")

    # --- 3. Compile and save the final GeoPackage file ---
    if not all_polygons:
        print("No boundaries were fetched. Exiting.")
        return

    print("Compiling all boundaries into a single GeoDataFrame...")
    # Create a GeoDataFrame from the list of Polygon objects
    final_gdf = gpd.GeoDataFrame(geometry=all_polygons, crs=TARGET_CRS)

    # Add the 'class' attribute required for the training script
    final_gdf['class'] = 'informal_settlement'
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(OUTPUT_ANNOTATION_FILE)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Saving final annotations to {OUTPUT_ANNOTATION_FILE}...")
    final_gdf.to_file(OUTPUT_ANNOTATION_FILE, driver="GPKG")

    print("--- Automated Data Generation Complete ---")
    print(f"Final dataset contains {len(final_gdf)} polygons.")

if __name__ == "__main__":
    main()
# scripts/00_acquire_aoi.py
import osmnx as ox
import geopandas as gpd
import os

# --- Configuration ---
# Define the query for the area of interest.
PLACE_QUERY = "Jakarta, Indonesia"

# Define the output path for the raw GeoJSON file.
OUTPUT_DIR = "data/raw"
OUTPUT_FILENAME = "jakarta_boundary.geojson"
OUTPUT_FILEPATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

# --- Main Execution ---
def main():
    """
    Fetches the administrative boundary for a specified place from OpenStreetMap
    and saves it as a GeoJSON file. This is the foundational first step for
    the data pipeline.
    """
    print(f"--- Starting AOI Acquisition for '{PLACE_QUERY}' ---")

    try:
        # --- 1. Fetch the GeoDataFrame for the specified place ---
        print(f"Querying OpenStreetMap for the boundary of '{PLACE_QUERY}'...")
        # 'geocode_to_gdf' retrieves the boundary polygon for the query.
        gdf = ox.geocode_to_gdf(PLACE_QUERY)
        print("Successfully retrieved boundary data.")

        # --- 2. Ensure the output directory exists ---
        print(f"Preparing to save file to {OUTPUT_FILEPATH}...")
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            print(f"Created directory: {OUTPUT_DIR}")

        # --- 3. Save the GeoDataFrame to GeoJSON ---
        gdf.to_file(OUTPUT_FILEPATH, driver='GeoJSON')
        print(f"Successfully saved Jakarta AOI to {OUTPUT_FILEPATH}")
        print("--- AOI Acquisition Complete ---")

    except Exception as e:
        print(f"\nAn error occurred during the acquisition process: {e}")
        print("Please check your internet connection and ensure the place query is valid.")

if __name__ == "__main__":
    main()
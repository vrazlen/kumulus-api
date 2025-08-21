import geopandas as gpd
import numpy as np
from shapely.geometry import box

# --- Configuration ---
INPUT_ANNOTATIONS_PATH = "data/processed/jakarta_annotations_multiclass.gpkg"
TRAINING_OUTPUT_PATH = "data/processed/training_set.gpkg"
VALIDATION_OUTPUT_PATH = "data/processed/validation_set.gpkg"

GRID_SIZE_METERS = 1000
VALIDATION_SPLIT_RATIO = 0.2
RANDOM_SEED = 42

def main():
    """
    Splits multi-class annotations into training and validation sets
    using a randomized spatial grid to ensure representative distribution.
    """
    print("--- Starting Randomized Spatial Dataset Split ---")

    gdf = gpd.read_file(INPUT_ANNOTATIONS_PATH)
    if gdf.empty:
        print("Input GeoDataFrame is empty. Exiting.")
        return

    total_bounds = gdf.total_bounds
    xmin, ymin, xmax, ymax = total_bounds

    grid_cells = [box(x, y, x + GRID_SIZE_METERS, y + GRID_SIZE_METERS) 
                  for x in np.arange(xmin, xmax, GRID_SIZE_METERS) 
                  for y in np.arange(ymin, ymax, GRID_SIZE_METERS)]
    
    grid_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs=gdf.crs)
    grid_gdf['grid_id'] = np.arange(len(grid_gdf))
    print(f"Created a grid with {len(grid_gdf)} cells.")

    joined_gdf = gpd.sjoin(gdf, grid_gdf, how="inner", predicate="intersects")
    unique_grid_ids = joined_gdf['grid_id'].unique()
    
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(unique_grid_ids)

    split_index = int(len(unique_grid_ids) * (1 - VALIDATION_SPLIT_RATIO))
    train_grid_ids, val_grid_ids = unique_grid_ids[:split_index], unique_grid_ids[split_index:]

    print(f"Randomly assigned {len(train_grid_ids)} grid cells to Training and {len(val_grid_ids)} to Validation.")

    train_gdf = joined_gdf[joined_gdf['grid_id'].isin(train_grid_ids)][['geometry', 'class']].copy()
    val_gdf = joined_gdf[joined_gdf['grid_id'].isin(val_grid_ids)][['geometry', 'class']].copy()

    print(f"  -> Generated {len(train_gdf)} training and {len(val_gdf)} validation polygons.")

    if not train_gdf.empty:
        train_gdf.to_file(TRAINING_OUTPUT_PATH, driver="GPKG")
    if not val_gdf.empty:
        val_gdf.to_file(VALIDATION_OUTPUT_PATH, driver="GPKG")
        
    print(f"--- Dataset split complete. Files saved. ---")

if __name__ == "__main__":
    main()
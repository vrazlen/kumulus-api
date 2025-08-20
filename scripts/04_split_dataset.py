# scripts/04_split_dataset.py
import geopandas as gpd
from sklearn.model_selection import train_test_split
import os

# --- Configuration ---
INPUT_ANNOTATIONS_PATH = "data/processed/jakarta_annotations_multiclass.gpkg"
OUTPUT_DIR = "data/processed"
TRAIN_SET_PATH = os.path.join(OUTPUT_DIR, "training_set.gpkg")
VALIDATION_SET_PATH = os.path.join(OUTPUT_DIR, "validation_set.gpkg")
VALIDATION_SPLIT_SIZE = 0.2 # 20% of the data will be used for validation
RANDOM_STATE = 42 # Ensures the split is the same every time we run it

def main():
    """
    Splits the main annotation GeoPackage into training and validation sets.
    """
    print("--- Starting Dataset Split ---")

    if not os.path.exists(INPUT_ANNOTATIONS_PATH):
        print(f"Error: Full annotation file not found at {INPUT_ANNOTATIONS_PATH}")
        return

    print(f"Reading full dataset from {INPUT_ANNOTATIONS_PATH}...")
    full_gdf = gpd.read_file(INPUT_ANNOTATIONS_PATH)
    print(f"Loaded {len(full_gdf)} polygons.")

    # --- Perform the 80/20 split ---
    # The 'train_test_split' function from scikit-learn is the industry
    # standard for this task.
    train_gdf, val_gdf = train_test_split(
        full_gdf,
        test_size=VALIDATION_SPLIT_SIZE,
        random_state=RANDOM_STATE
    )
    print(f"Splitting data: {len(train_gdf)} for training, {len(val_gdf)} for validation.")

    # --- Save the new datasets ---
    print(f"Saving training set to {TRAIN_SET_PATH}...")
    train_gdf.to_file(TRAIN_SET_PATH, driver="GPKG")

    print(f"Saving validation set to {VALIDATION_SET_PATH}...")
    val_gdf.to_file(VALIDATION_SET_PATH, driver="GPKG")

    print("--- Dataset Split Complete ---")

if __name__ == "__main__":
    main()
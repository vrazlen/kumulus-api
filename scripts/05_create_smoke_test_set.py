# scripts/05_create_smoke_test_set.py
import os
import glob
import shutil
from pathlib import Path

print("--- Creating Smoke Test Dataset ---")

# --- Configuration ---
SOURCE_DIR = Path("data/processed")
TARGET_DIR = Path("data/smoke_test")
IMAGE_DIR = TARGET_DIR / "patches"  # Corrected subdirectory
MASK_DIR = TARGET_DIR / "masks"
NUM_FILES_TO_COPY = 16

def create_smoke_test_set():
    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)
    IMAGE_DIR.mkdir(parents=True)
    MASK_DIR.mkdir(parents=True)

    print(f"Copying {NUM_FILES_TO_COPY} files from {SOURCE_DIR} to {TARGET_DIR}...")
    source_images = sorted(glob.glob(str(SOURCE_DIR / "patches" / "*.tif")))
    
    if len(source_images) < NUM_FILES_TO_COPY:
        raise ValueError(f"Source directory 'data/processed/patches' has fewer than {NUM_FILES_TO_COPY} images.")

    copied_count = 0
    for source_img_path_str in source_images[:NUM_FILES_TO_COPY]:
        source_img_path = Path(source_img_path_str)
        mask_filename = source_img_path.name
        source_mask_path = SOURCE_DIR / "masks" / mask_filename
        
        if source_mask_path.exists():
            shutil.copy(source_img_path, IMAGE_DIR / mask_filename)
            shutil.copy(source_mask_path, MASK_DIR / mask_filename)
            copied_count += 1
        else:
            print(f"Warning: Mask not found for {source_img_path.name}, skipping.")
            
    print(f"Successfully created smoke test set with {copied_count} image/mask pairs.")
    print("Dataset located at:", TARGET_DIR)

if __name__ == "__main__":
    create_smoke_test_set()
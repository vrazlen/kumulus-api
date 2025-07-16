# src/train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm
import os

# Import our custom dataset and transforms
from src.dataset import SegmentationDataset, get_train_transform

print("--- Training script started. Loading libraries... ---")

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4 # Adjusted for memory constraints
EPOCHS = 25
LEARNING_RATE = 1e-4
MODEL_OUTPUT_DIR = 'models'
MODEL_NAME = 'unet_slum_segmentation.pth'

def main():
    """
    Main function to run the model training pipeline.
    """
    print(f"Using device: {DEVICE}")

    # --- CORRECTED DATASET INITIALIZATION ---
    # The dataset class now knows its own paths.
    dataset = SegmentationDataset(transform=get_train_transform())

    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0 # Set to 0 for Windows compatibility
    )

    # Initialize the U-Net model
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet", # Use pre-trained weights for better feature extraction
        in_channels=3,
        classes=1,
    ).to(DEVICE)

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training Loop ---
    print("--- Starting Training ---")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, masks in progress_bar:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.4f}")

    # --- Save the Model ---
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_OUTPUT_DIR, MODEL_NAME)
    torch.save(model.state_dict(), model_path)
    print(f"Training complete. Model saved to {model_path}")

if __name__ == '__main__':
    main()
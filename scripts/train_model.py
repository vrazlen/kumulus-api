import os
import numpy as np
import torch
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import albumentations as A
from albumentations.pytorch import ToTensorV2
import mlflow
import mlflow.pytorch
from sklearn.metrics import precision_score, recall_score, jaccard_score

# --- 1. Configuration, Hyperparameters & MLflow Setup ---
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "KUMULUS_Settlement_Detection_v2"

HPARAMS = {
    "encoder_name": "resnet50",
    "learning_rate": 1e-4,
    "batch_size": 8,
    "epochs": 50, # INCREASED EPOCHS
    "chip_size": 256,
    "loss_function": "WeightedCrossEntropyLoss",
}

INPUT_IMAGE_PATH = "data/processed/jakarta_sentinel_clipped_4band.tif"
TRAIN_LABELS_PATH = "data/processed/training_set.gpkg"
VAL_LABELS_PATH = "data/processed/validation_set.gpkg"

CLASSES = ['background', 'informal_settlement', 'water', 'vegetation']
CLASS_MAP = {name: i for i, name in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

# --- 2. Data Loading & Augmentation ---
def get_train_augs():
    return A.Compose([
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
        A.ColorJitter(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_val_augs():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def rasterize_multiclass_labels(image_path, labels_path):
    print(f"Rasterizing multi-class labels from {os.path.basename(labels_path)}...")
    with rasterio.open(image_path) as src:
        meta, transform, out_shape = src.meta.copy(), src.transform, src.shape
    labels_gdf = gpd.read_file(labels_path)
    if labels_gdf.crs != meta['crs']: labels_gdf = labels_gdf.to_crs(meta['crs'])
    shapes_with_values = [(geom, CLASS_MAP.get(name, 0)) for geom, name in zip(labels_gdf.geometry, labels_gdf['class'])]
    mask = rasterize(shapes=shapes_with_values, out_shape=out_shape, transform=transform, fill=CLASS_MAP['background'], dtype=rasterio.uint8)
    print("Rasterization complete.")
    return mask

class SegmentationDataset(Dataset):
    def __init__(self, image, mask, chip_size, augmentations=None):
        self.image, self.mask, self.chip_size, self.augmentations = image, mask, chip_size, augmentations
        self.chips = [(x, y) for y in range(0, image.shape[1] - chip_size + 1, chip_size) for x in range(0, image.shape[2] - chip_size + 1, chip_size)]
        print(f"Dataset initialized with {len(self.chips)} chips.")
    def __len__(self): return len(self.chips)
    def __getitem__(self, idx):
        x, y = self.chips[idx]
        image_chip = np.transpose(self.image[:3, y:y+self.chip_size, x:x+self.chip_size], (1, 2, 0))
        mask_chip = self.mask[y:y+self.chip_size, x:x+self.chip_size]
        if self.augmentations:
            augmented = self.augmentations(image=image_chip, mask=mask_chip)
            image_chip, mask_chip = augmented['image'], augmented['mask']
        return image_chip, mask_chip.long()

def calculate_metrics(pred, target):
    pred_classes = torch.argmax(pred, dim=1).cpu().numpy().flatten()
    target_classes = target.cpu().numpy().flatten()
    settlement_id = CLASS_MAP['informal_settlement']
    iou = jaccard_score(target_classes, pred_classes, labels=[settlement_id], average='micro', zero_division=0)
    precision = precision_score(target_classes, pred_classes, labels=[settlement_id], average='micro', zero_division=0)
    recall = recall_score(target_classes, pred_classes, labels=[settlement_id], average='micro', zero_division=0)
    return iou, precision, recall

def train_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run() as run:
        print(f"Starting MLflow Run: {run.info.run_id}")
        mlflow.log_params(HPARAMS)
        
        with rasterio.open(INPUT_IMAGE_PATH) as src:
            def scale_band(band):
                min_v, max_v = np.percentile(band, [2, 98])
                if max_v == min_v: return np.zeros_like(band, dtype=np.uint8)
                return np.clip((band - min_v) / (max_v - min_v) * 255.0, 0, 255).astype(np.uint8)
            image_data = np.array([scale_band(b) for b in src.read()])

        train_mask = rasterize_multiclass_labels(INPUT_IMAGE_PATH, TRAIN_LABELS_PATH)
        val_mask = rasterize_multiclass_labels(INPUT_IMAGE_PATH, VAL_LABELS_PATH)
        
        print("Calculating class weights for loss function...")
        counts = np.bincount(train_mask.flatten(), minlength=NUM_CLASSES)
        weights = counts.sum() / (NUM_CLASSES * counts + 1e-6)
        class_weights_tensor = torch.tensor(weights, dtype=torch.float)
        print(f"Calculated Class Weights: {weights}")
        mlflow.log_param("class_weights", weights.tolist())

        train_ds = SegmentationDataset(image_data, train_mask, HPARAMS["chip_size"], get_train_augs())
        train_dl = DataLoader(train_ds, batch_size=HPARAMS["batch_size"], shuffle=True)
        val_ds = SegmentationDataset(image_data, val_mask, HPARAMS["chip_size"], get_val_augs())
        val_dl = DataLoader(val_ds, batch_size=HPARAMS["batch_size"], shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = smp.DeepLabV3Plus(encoder_name=HPARAMS["encoder_name"], encoder_weights="imagenet", in_channels=3, classes=NUM_CLASSES).to(device)
        loss_fn = CrossEntropyLoss(weight=class_weights_tensor.to(device))
        optimizer = AdamW(model.parameters(), lr=HPARAMS["learning_rate"])
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

        best_iou = 0.0
        for epoch in range(HPARAMS["epochs"]):
            model.train()
            train_loss = 0.0
            for imgs, masks in tqdm(train_dl, desc=f"Epoch {epoch+1} [Train]"):
                imgs, masks = imgs.to(device), masks.to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = loss_fn(outputs, masks)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            model.eval()
            iou, precision, recall = 0, 0, 0
            with torch.no_grad():
                for imgs, masks in tqdm(val_dl, desc=f"Epoch {epoch+1} [Validate]"):
                    imgs, masks = imgs.to(device), masks.to(device)
                    outputs = model(imgs)
                    i, p, r = calculate_metrics(outputs, masks)
                    iou += i; precision += p; recall += r

            avg_iou = iou / len(val_dl)
            avg_prec = precision / len(val_dl)
            avg_recall = recall / len(val_dl)
            
            metrics = {
                "val_iou": avg_iou, "val_precision": avg_prec, 
                "val_recall": avg_recall, "train_loss": train_loss / len(train_dl)
            }
            mlflow.log_metrics(metrics, step=epoch)
            print(f"Epoch {epoch+1} >> IoU: {avg_iou:.4f} | Precision: {avg_prec:.4f} | Recall: {avg_recall:.4f}")

            scheduler.step(avg_iou)

            if avg_iou > best_iou:
                best_iou = avg_iou
                mlflow.pytorch.log_model(model, "model", registered_model_name="KUMULUS-Segmentation-Model-v2")
                print(f"  -> New best model saved as MLflow artifact.")
        print("\n--- Training Complete ---")

if __name__ == "__main__":
    train_model()
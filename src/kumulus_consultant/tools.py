import json
import logging
import os
from typing import Dict, Any

import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from PIL import Image
from torchvision import transforms
from rasterio.features import shapes
from langchain_core.tools import tool
from pydantic.v1 import BaseModel, Field
from tqdm import tqdm
from sentinelhub import (
    SHConfig, SentinelHubRequest, DataCollection, MimeType, bbox_to_dimensions
)

from kumulus_consultant.config.settings import settings

# --- Module-level Logger Setup ---
logger = logging.getLogger(__name__)

# --- Custom Exceptions & Pydantic Schemas ---
class ToolError(Exception):
    """Base class for tool-specific errors."""
    pass
class DataUnavailableError(ToolError):
    """Raised when required data cannot be found or accessed."""
    pass
class APIFailureError(ToolError):
    """Raised when an external API call fails."""
    pass
class ModelInferenceError(ToolError):
    """Raised when the ML model fails during inference."""
    pass
class AnalyzeNDVIInput(BaseModel):
    """Input model for the analyze_green_space_ndvi tool."""
    location_geotiff_path: str = Field(description="The local file path to the 4-band GeoTIFF file.")
class FetchSentinelInput(BaseModel):
    """Input model for the fetch_sentinel2_data tool."""
    latitude: float = Field(description="Latitude of the center point of the area of interest.")
    longitude: float = Field(description="Longitude of the center point of the area of interest.")
    start_date: str = Field(description="Start date for the imagery search in YYYY-MM-DD format.")
    end_date: str = Field(description="End date for the imagery search in YYYY-MM-DD format.")
    size_meters: int = Field(default=1000, description="The width and height of the square area to fetch in meters.")
class DetectSettlementsInput(BaseModel):
    """Input schema for the detect_informal_settlements tool."""
    processed_image_path: str = Field(description="Path to the processed, clipped satellite image GeoTIFF.")

# --- Helper Functions for Inference ---
def _load_model(model_path, device):
    """Loads the pre-trained DeepLabv3+ model."""
    logger.debug(f"Loading model from {model_path} onto device {device}.")
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1, # Binary segmentation
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def _preprocess_image(image_array_hwc):
    """
    Takes a numpy array (H, W, C), pads it to be divisible by 16,
    and applies PyTorch transformations.
    """
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        image = Image.fromarray(image_array_hwc, 'RGB')
        tensor = preprocess(image).unsqueeze(0)

        _, _, h, w = tensor.shape
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16

        padded_tensor = F.pad(tensor, (0, pad_w, 0, pad_h), "constant", 0)
        return padded_tensor, (pad_h, pad_w)

    except Exception as e:
        logger.error(f"Failed during image preprocessing. Reason: {e}")
        raise

# --- Production-Ready LangChain Tools ---
@tool(args_schema=AnalyzeNDVIInput)
def analyze_green_space_ndvi(location_geotiff_path: str) -> str:
    """Analyzes green space and vegetation health from a 4-band GeoTIFF file."""
    logger.info(f"Starting NDVI analysis for: {location_geotiff_path}")
    try:
        with rasterio.open(location_geotiff_path) as src:
            if src.count < 4:
                raise ValueError("Input GeoTIFF must have at least 4 bands (B, G, R, NIR).")

            logger.debug("Reading Red (Band 3) and NIR (Band 4) from stacked file.")
            band_red = src.read(3).astype(float)
            band_nir = src.read(4).astype(float)

            np.seterr(divide='ignore', invalid='ignore')
            denominator = band_nir + band_red
            ndvi = np.where(denominator == 0, 0, (band_nir - band_red) / denominator)

            average_ndvi = np.nanmean(ndvi)

            logger.info(f"Successfully calculated average NDVI: {average_ndvi:.4f}")

            result = {
                "average_ndvi": round(average_ndvi, 4),
                "summary": (
                    "Vegetation health is moderate to high."
                    if average_ndvi > 0.3
                    else "Vegetation is sparse or stressed."
                ),
            }
            response = {"status": "success", "data": result}
            return json.dumps(response, indent=2)

    except rasterio.errors.RasterioIOError as e:
        error_message = f"Could not read or find the GeoTIFF file at '{location_geotiff_path}'. Reason: {e}"
        logger.error(error_message)
        raise DataUnavailableError(error_message) from e

    except Exception as e:
        error_message = f"An unexpected error occurred during NDVI analysis. Reason: {e}"
        logger.exception(error_message)
        response = {"status": "error", "message": error_message}
        return json.dumps(response, indent=2)

@tool(args_schema=FetchSentinelInput)
def fetch_sentinel2_data(latitude: float, longitude: float, start_date: str, end_date: str, size_meters: int = 1000) -> str:
    """
    Fetches a Sentinel-2 L2A GeoTIFF for a given location, date, and area size.
    """
    logger.info(f"Initiating Sentinel-2 data fetch for coordinates ({latitude}, {longitude}) between {start_date} and {end_date}.")
    try:
        config = SHConfig()
        config.sh_client_id = os.getenv("SH_CLIENT_ID")
        config.sh_client_secret = os.getenv("SH_CLIENT_SECRET")
        if not config.sh_client_id or not config.sh_client_secret:
            raise APIFailureError("Sentinel Hub credentials (SH_CLIENT_ID, SH_CLIENT_SECRET) are not configured.")

        deg_per_meter = 1 / 111320.0
        size_deg = size_meters * deg_per_meter
        bbox_wgs84 = (longitude - size_deg / 2, latitude - size_deg / 2, longitude + size_deg / 2, latitude + size_deg / 2)
        aoi_bbox = rasterio.coords.BoundingBox(*bbox_wgs84)
        aoi_dims = bbox_to_dimensions(aoi_bbox, resolution=10)
        
        evalscript = """
        //VERSION=3
        function setup() {{ return {{ input: ["B02", "B03", "B04", "B08", "dataMask"], output: {{ bands: 5, sampleType: "UINT16" }} }}; }}
        function evaluatePixel(sample) {{ return [sample.B02, sample.B03, sample.B04, sample.B08, sample.dataMask]; }}
        """
        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A.with_cloud_coverage(30),
                time_interval=(start_date, end_date)
            )],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=aoi_bbox,
            size=aoi_dims,
            config=config,
        )

        logger.info("Sending request to Sentinel Hub API...")
        image_data_list = request.get_data(save_data=False)

        if not image_data_list:
            raise DataUnavailableError(f"No Sentinel-2 imagery found for the specified location and date range ({start_date} to {end_date}).")

        output_dir = os.path.join("data", "raw")
        os.makedirs(output_dir, exist_ok=True)
        file_name = f"sentinel2_{latitude}_{longitude}_{start_date}.tif"
        file_path = os.path.join(output_dir, file_name)
        
        with open(file_path, "wb") as f:
            f.write(image_data_list[0])
            
        logger.info(f"Successfully downloaded and saved Sentinel-2 image to {file_path}")
        return json.dumps({"status": "success", "file_path": file_path})

    except (APIFailureError, DataUnavailableError) as e:
        logger.error(str(e))
        return json.dumps({"status": "error", "message": str(e)})
    except Exception as e:
        logger.exception("An unexpected error occurred while fetching Sentinel-2 data.")
        return json.dumps({"status": "error", "message": f"An unexpected error occurred: {e}"})

@tool(args_schema=DetectSettlementsInput)
def detect_informal_settlements(processed_image_path: str) -> str:
    """
    Performs semantic segmentation on a large satellite image using a tiling
    strategy. It detects informal settlements, vectorizes the results,
    and saves them to a GeoPackage file.
    """
    logger.info(f"Starting tiled settlement detection for: {processed_image_path}")
    
    TILE_SIZE = 512
    
    try:
        # --- 1. Load Model and Setup ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _load_model(settings.MODEL_PATH, device)

        with rasterio.open(processed_image_path) as src:
            image_height, image_width = src.shape
            full_prediction_mask = np.zeros((image_height, image_width), dtype=np.uint8)
            
            # --- 2. Tiling Inference Loop ---
            logger.info(f"Processing image in {TILE_SIZE}x{TILE_SIZE} tiles...")
            x_coords = range(0, image_width, TILE_SIZE)
            y_coords = range(0, image_height, TILE_SIZE)
            tile_iterator = tqdm(total=len(x_coords) * len(y_coords), desc="Tiled Inference Progress")
            
            for y in y_coords:
                for x in x_coords:
                    window = Window(x, y, TILE_SIZE, TILE_SIZE)
                    tile_data = src.read([1, 2, 3], window=window)
                    if tile_data.max() == 0: 
                        tile_iterator.update(1)
                        continue
                    
                    def scale_band(band):
                        min_val, max_val = np.percentile(band, [2, 98])
                        if max_val == min_val: return np.zeros_like(band, dtype=np.uint8)
                        scaled = (band - min_val) / (max_val - min_val) * 255.0
                        return np.clip(scaled, 0, 255).astype(np.uint8)

                    tile_uint8 = np.array([scale_band(band) for band in tile_data])
                    tile_hwc = np.transpose(tile_uint8, (1, 2, 0))
                    padded_tensor, _ = _preprocess_image(tile_hwc)
                    
                    with torch.no_grad():
                        prediction = model(padded_tensor.to(device))
                        prediction = torch.sigmoid(prediction).cpu().numpy()[0, 0, :, :]
                    
                    h_unpadded, w_unpadded = tile_hwc.shape[:2]
                    prediction = prediction[:h_unpadded, :w_unpadded]
                    tile_mask = (prediction > 0.5).astype(np.uint8)
                    h, w = tile_mask.shape
                    full_prediction_mask[y:y+h, x:x+w] = tile_mask
                    tile_iterator.update(1)
            
            tile_iterator.close()
            
            # --- 3. Vectorize Full Mask ---
            image_transform = src.transform
            image_crs = src.crs
            mask_shapes = shapes(full_prediction_mask, mask=(full_prediction_mask == 1), transform=image_transform)
            features = [{"type": "Feature", "geometry": geom, "properties": {"prediction_class": "informal_settlement"}} for geom, val in mask_shapes if val == 1]

            if not features:
                logger.warning("No informal settlements detected.")
                return json.dumps({"status": "success", "data": None, "message": "No informal settlements were detected."})

            # --- 4. Save Results to GeoPackage File ---
            logger.info(f"Vectorized prediction into {len(features)} features.")
            gdf = gpd.GeoDataFrame.from_features(features, crs=image_crs)
            
            output_dir = os.path.join("data", "output")
            os.makedirs(output_dir, exist_ok=True)
            output_file_path = os.path.join(output_dir, "detected_settlements.gpkg")

            gdf.to_file(output_file_path, driver="GPKG")
            logger.info(f"Successfully saved {len(gdf)} features to {output_file_path}")

            # --- 5. Update Success Response ---
            return json.dumps({
                "status": "success",
                "data": {
                    "output_file_path": output_file_path,
                    "detected_features_count": len(gdf)
                },
                "message": f"Successfully detected {len(gdf)} settlement areas and saved the results to {output_file_path}."
            })

    except (ModelInferenceError, DataUnavailableError) as e:
        error_message = f"An error occurred during settlement detection: {e}"
        logger.error(error_message)
        return json.dumps({"status": "error", "message": error_message})
    except Exception as e:
        error_message = f"An unexpected error occurred during tiled settlement detection: {e}"
        logger.exception(error_message)
        raise ModelInferenceError(error_message) from e
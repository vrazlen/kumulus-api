# src/kumulus_consultant/tools.py
import json
import logging
import os
from typing import Dict, Any

import numpy as np
import rasterio
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from sentinelhub import (
    SHConfig,
    SentinelHubRequest,
    DataCollection,
    MimeType,
    bbox_to_dimensions,
)

# --- Module-level Logger Setup ---
logger = logging.getLogger(__name__)

# --- Custom Exceptions ---
class ToolError(Exception): pass
class DataUnavailableError(ToolError): pass
class APIFailureError(ToolError): pass

# --- Pydantic Input Schema for our Tool ---
class AnalyzeNDVIInput(BaseModel):
    """Input model for the analyze_green_space_ndvi tool."""
    location_geotiff_path: str = Field(description="The local file path to the 4-band GeoTIFF file.")

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

@tool
def fetch_sentinel2_data(latitude: float, longitude: float, start_date: str, end_date: str, size_meters: int = 1000) -> str:
    """Fetches a Sentinel-2 L2A GeoTIFF for a given location, date, and area size."""
    logger.info(f"Fetching Sentinel-2 data for ({latitude}, {longitude})")
    config = SHConfig()
    if not config.sh_client_id or not config.sh_client_secret:
        config.sh_client_id = os.getenv("SH_CLIENT_ID")
        config.sh_client_secret = os.getenv("SH_CLIENT_SECRET")
        if not config.sh_client_id or not config.sh_client_secret:
            error_message = "Sentinel Hub credentials (SH_CLIENT_ID, SH_CLIENT_SECRET) are missing."
            return json.dumps({"status": "error", "message": error_message})
    
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
    try:
        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L2A.with_cloud_coverage(30), time_interval=(start_date, end_date))],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=aoi_bbox, size=aoi_dims, config=config,
        )
        image_data = request.get_data(save_data=False)[0]
        file_name = f"sentinel2_{latitude}_{longitude}_{start_date}.tif"
        file_path = os.path.join("data", "raw", file_name)
        with open(file_path, "wb") as f: f.write(image_data)
        logger.info(f"Successfully downloaded and saved image to {file_path}")
        return json.dumps({{"status": "success", "file_path": file_path}})
    except Exception as e:
        error_message = f"An error occurred fetching data from Sentinel Hub: {e}"
        logger.exception(error_message)
        return json.dumps({{"status": "error", "message": error_message}})
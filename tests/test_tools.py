# tests/test_tools.py
import json
from unittest.mock import MagicMock
import numpy as np
import pytest
from rasterio.errors import RasterioIOError

# Import the specific tool and custom exception we need to test
from kumulus_consultant.tools import (
    analyze_green_space_ndvi,
    DataUnavailableError,
)

@pytest.fixture
def mock_rasterio_open_success(mocker):
    """
    Pytest fixture to mock rasterio.open for a successful read.

    This mock simulates the context manager (`with...as`) behavior of
    rasterio.open and makes the `read()` method return predefined NumPy arrays
    for the Red and NIR bands.
    """
    # Create a mock dataset reader object
    mock_dataset = MagicMock()
    
    # Configure the read method to return different arrays based on the band index
    # Band 4 (Red) will be a 2x2 array of 0.1s
    # Band 8 (NIR) will be a 2x2 array of 0.8s
    mock_dataset.read.side_effect = [
        np.full((2, 2), 0.1, dtype=float),  # Corresponds to src.read(4)
        np.full((2, 2), 0.8, dtype=float),  # Corresponds to src.read(8)
    ]
    
    # The mock needs to support the `with` statement (context manager protocol)
    mock_context_manager = MagicMock()
    mock_context_manager.__enter__.return_value = mock_dataset
    mock_context_manager.__exit__.return_value = None

    # Patch 'rasterio.open' specifically within the 'tools' module where it is called
    return mocker.patch(
        "kumulus_consultant.tools.rasterio.open", return_value=mock_context_manager
    )

@pytest.fixture
def mock_rasterio_open_failure(mocker):
    """
    Pytest fixture to mock rasterio.open to simulate a file-not-found error.
    This mock will raise a RasterioIOError when called.
    """
    return mocker.patch(
        "kumulus_consultant.tools.rasterio.open",
        side_effect=RasterioIOError("File not found."),
    )

def test_analyze_green_space_ndvi_success(mock_rasterio_open_success):
    """
    Tests the success case for the analyze_green_space_ndvi tool.

    Asserts that when rasterio provides valid data, the tool calculates the
    correct average NDVI and returns a standardized JSON success response.
    """
    # GIVEN a valid file path (the mock will intercept the actual file open)
    file_path = "/fake/path/to/image.tif"

    # WHEN the tool is called
    result_json = analyze_green_space_ndvi(file_path)

    # THEN the result should be a valid JSON string indicating success
    result_data = json.loads(result_json)
    assert result_data["status"] == "success"
    assert "data" in result_data

    # AND the average NDVI should be calculated correctly
    # For Red=0.1 and NIR=0.8, NDVI = (0.8 - 0.1) / (0.8 + 0.1) = 0.7 / 0.9
    expected_ndvi = 0.7 / 0.9
    assert result_data["data"]["average_ndvi"] == pytest.approx(expected_ndvi, 0.0001)

    # AND the summary should reflect high vegetation health
    assert "high" in result_data["data"]["summary"]

    # AND the mock for rasterio.open was called exactly once with the correct path
    mock_rasterio_open_success.assert_called_once_with(file_path)

def test_analyze_green_space_ndvi_failure_raises_custom_exception(
    mock_rasterio_open_failure,
):
    """
    Tests the failure case for the analyze_green_space_ndvi tool.

    Asserts that when rasterio.open raises a RasterioIOError, the tool
    correctly catches it and raises our custom DataUnavailableError.
    """
    # GIVEN an invalid file path that will cause our mock to raise an error
    file_path = "/invalid/path/image.tif"

    # WHEN the tool is called within a pytest.raises context manager
    # THEN it should raise our specific custom exception
    with pytest.raises(DataUnavailableError) as exc_info:
        analyze_green_space_ndvi(file_path)

    # AND the exception message should contain the original reason for clarity
    assert "Could not read or find the GeoTIFF file" in str(exc_info.value)
    assert "File not found" in str(exc_info.value)

    # AND the mock for rasterio.open was called exactly once
    mock_rasterio_open_failure.assert_called_once_with(file_path)
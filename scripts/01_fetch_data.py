# scripts/01_fetch_data.py
from dotenv import load_dotenv
load_dotenv()
import os
import requests
import geopandas as gpd
import pandas as pd
from tqdm import tqdm

# --- Configuration ---
COPERNICUS_USER = os.getenv('COPERNICUS_USER')
COPERNICUS_PASSWORD = os.getenv('COPERNICUS_PASSWORD')

# Modern API endpoints
CATALOG_API_URL = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
ODATA_SEARCH_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
# DEFINITIVE CORRECTION: Use the 'zipper' service endpoint AND remove single quotes around the UUID
ZIPPER_DOWNLOAD_URL_TEMPLATE = "https://zipper.dataspace.copernicus.eu/odata/v1/Products({uuid})/$value"

AOI_FILE = 'data/raw/jakarta_aoi_correct.geojson'
OUTPUT_DIR = 'data/raw/sentinel'

# --- Main Logic ---
def get_auth_token():
    """Gets an authentication token for the Copernicus APIs."""
    token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    auth_data = {
        "client_id": "cdse-public", "username": COPERNICUS_USER,
        "password": COPERNICUS_PASSWORD, "grant_type": "password",
    }
    response = requests.post(token_url, data=auth_data)
    response.raise_for_status()
    return response.json()["access_token"]

def download_file(url, output_path, headers):
    """Downloads a large file with a progress bar."""
    print(f"Downloading from Zipper service: {url}")
    with requests.get(url, headers=headers, stream=True, allow_redirects=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        with open(output_path, 'wb') as f, tqdm(
            total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(output_path)
        ) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    print("\nDownload complete.")

def fetch_best_sentinel_scene():
    """
    Uses a two-step search: Catalog API to find the name, OData API to get the UUID,
    and the Zipper service to download the product.
    """
    print("Getting Copernicus authentication token...")
    try:
        token = get_auth_token()
    except requests.exceptions.HTTPError as e:
        print(f"Authentication failed: {e}"); return
    headers = {"Authorization": f"Bearer {token}"}

    print(f"Loading AOI from {AOI_FILE}...")
    aoi_gdf = gpd.read_file(AOI_FILE)
    bbox = list(aoi_gdf.total_bounds)

    print("Step 1: Querying Catalog API for best scene NAME...")
    search_payload = { "bbox": bbox, "datetime": "2023-01-01T00:00:00Z/2023-12-31T23:59:59Z", "collections": ["sentinel-2-l2a"], "limit": 100 }
    response = requests.post(CATALOG_API_URL, headers=headers, json=search_payload)
    response.raise_for_status()
    results = response.json()
    if not results['features']: print("No products found."); return

    features_df = pd.json_normalize(results['features'])
    df_sorted = features_df.sort_values(by='properties.eo:cloud_cover', ascending=True)
    best_scene = df_sorted.iloc[0]
    
    product_name = best_scene['id']
    cloud_cover = best_scene['properties.eo:cloud_cover']
    print(f"Best scene found: {product_name} with {cloud_cover}% cloud cover.")

    print(f"\nStep 2: Querying OData API for product UUID...")
    odata_filter = f"Name eq '{product_name}'"
    odata_params = {"$filter": odata_filter, "$select": "Id"}
    response = requests.get(ODATA_SEARCH_URL, headers=headers, params=odata_params)
    response.raise_for_status()
    odata_results = response.json()
    if not odata_results['value']: print("Could not find product in OData catalog."); return
    
    product_uuid = odata_results['value'][0]['Id']
    print(f"Found Product UUID: {product_uuid}")

    print("\nStep 3: Starting download...")
    download_url = ZIPPER_DOWNLOAD_URL_TEMPLATE.format(uuid=product_uuid)
    file_name = f"{product_name}.zip"
    output_path = os.path.join(OUTPUT_DIR, file_name)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    download_file(download_url, output_path, headers)

if __name__ == '__main__':
    if not COPERNICUS_USER or not COPERNICUS_PASSWORD:
        print("Error: Set COPERNICUS_USER and COPERNICUS_PASSWORD environment variables.")
    else:
        fetch_best_sentinel_scene()
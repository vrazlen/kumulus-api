# app/dashboard.py
import streamlit as st
from PIL import Image
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="KUMULUS: Informal Settlement Identification",
    page_icon="🛰️",
    layout="wide"
)

# --- Main Application ---
st.title("KUMULUS: AI-Powered Informal Settlement Identification")

st.markdown("""
Welcome to the KUMULUS prototype dashboard. This tool uses a U-Net deep learning model 
to identify and segment informal settlements from Sentinel-2 satellite imagery. This demonstration
showcases the core functionality of our end-to-end data pipeline and machine learning model.
""")

# --- Asset Loading ---
ASSET_DIR = 'app/assets'
DEMO_DISTRICTS = ["District A", "District B", "District C"]

# --- Sidebar for Navigation ---
st.sidebar.title("Controls")
selected_district = st.sidebar.selectbox(
    "Select a District to Analyze",
    DEMO_DISTRICTS
)

# --- Image Display ---
st.header(f"Analysis for {selected_district}")

district_key = selected_district.lower().replace(' ', '_')
base_image_path = os.path.join(ASSET_DIR, f"{district_key}_base.png")
overlay_image_path = os.path.join(ASSET_DIR, f"{district_key}_overlay.png")

if os.path.exists(base_image_path) and os.path.exists(overlay_image_path):
    base_image = Image.open(base_image_path)
    overlay_image = Image.open(overlay_image_path)

    col1, col2 = st.columns(2)

    with col1:
        st.image(base_image, caption=f"Base Satellite Image: {selected_district}", use_column_width=True)

    with col2:
        st.image(overlay_image, caption=f"Model Prediction Overlay (in Red)", use_column_width=True)
else:
    st.error(f"Asset files for {selected_district} not found. Please ensure '04_generate_demo_assets.py' was run successfully.")


# --- Ethical Considerations Section ---
with st.expander("⚠️ Ethical Considerations & Model Limitations"):
    st.markdown("""
    As outlined in our project's guiding principles, it is critical to use this technology responsibly.
    * **Purpose:** This tool is designed for analysis and planning to support equitable urban development, not for surveillance or punitive action.
    * **Accuracy:** The model's predictions are based on visual patterns and are not a definitive ground truth. They are subject to errors and should be field-verified by local experts.
    * **Bias:** The training data is from a specific region and time. The model's performance may vary significantly in different geographic contexts or seasons.
    * **Community Engagement:** True validation and responsible use require partnership with the communities depicted in this data. This dashboard is a starting point for dialogue, not a final judgment.
    """)
# app.py
import streamlit as st
import pydeck as pdk
import pandas as pd
from src.kumulus_consultant.main import initialize_system, get_agent_response

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="KUMULUS AI Geo-Consultant",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🛰️ KUMULUS AI Geo-Consultant")
st.caption("An AI agent for interactive geospatial analysis.")

# --- 2. Agent and State Initialization ---
# Initialize the agent executor once and store it in session state.
if "agent_executor" not in st.session_state:
    with st.spinner("Initializing AI Agent... This may take a moment."):
        # We don't need the RAG system object in the UI, just the executor
        st.session_state.agent_executor, st.session_state.rag_system = initialize_system()

# Initialize chat history.
if "messages" not in st.session_state:
    st.session_state.messages = []

def render_map(geojson_data):
    """Renders a Pydeck map from GeoJSON data."""
    try:
        # Create a DataFrame to easily calculate the centroid for the map view
        features = geojson_data.get("features", [])
        if not features:
            st.warning("Geospatial data was generated but contained no features to display.")
            return

        coords = [f["geometry"]["coordinates"] for f in features]
        # Simplistic centroid calculation for initial view
        avg_lon = pd.Series([c[0][0][0] for c in coords]).mean()
        avg_lat = pd.Series([c[0][0][1] for c in coords]).mean()
        
        initial_view_state = pdk.ViewState(
            latitude=avg_lat,
            longitude=avg_lon,
            zoom=12,
            pitch=45,
            bearing=0
        )

        geojson_layer = pdk.Layer(
            "GeoJsonLayer",
            data=geojson_data,
            opacity=0.6,
            stroked=True,
            filled=True,
            get_fill_color="[255, 0, 0, 140]",  # Semi-transparent red
            get_line_color="[255, 0, 0, 255]",
            get_line_width=100,
            pickable=True
        )

        deck = pdk.Deck(
            map_style="mapbox://styles/mapbox/satellite-streets-v12",
            initial_view_state=initial_view_state,
            layers=[geojson_layer],
            tooltip={"text": "Detected Area: Informal Settlement"}
        )
        st.pydeck_chart(deck)
    except Exception as e:
        st.error(f"Failed to render map from geospatial data. Error: {e}")

# --- 3. Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "geojson" in message and message["geojson"]:
            render_map(message["geojson"])

# --- 4. User Input and Agent Invocation ---
if prompt := st.chat_input("Ask about geospatial analysis, e.g., 'Detect informal settlements near -6.99, 110.42'"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("KUMULUS is thinking..."):
            # Call the backend agent function
            agent_response = get_agent_response(
                prompt,
                st.session_state.agent_executor,
                st.session_state.rag_system
            )
            
            # Format the full textual response
            recommendation = agent_response.get("recommendation", "No recommendation available.")
            justification = agent_response.get("justification", "No justification provided.")
            ethical_check = agent_response.get("ethical_check", {})
            reason = ethical_check.get("reason", "Ethical check not performed.")
            
            full_response_text = f"""
            **Recommendation:**\n
            {recommendation}\n
            **Justification:**\n
            {justification}\n
            **Ethical Check:**\n
            > {reason}
            """
            message_placeholder.markdown(full_response_text)
            
            # Render map if GeoJSON data is present
            geojson_data = agent_response.get("geojson_data")
            if geojson_data:
                render_map(geojson_data)

            # Add the complete response to session state for history
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response_text,
                "geojson": geojson_data
            })
import streamlit as st
import pandas as pd
import osmnx as ox
import networkx as nx
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import random
import geopy.distance

# --- Streamlit Title ---
st.title("🚗 Chennai Smart Parking Route Finder with Data Extraction")

# --- File Upload ---
st.sidebar.header("📂 Upload Parking Data (CSV)")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

# --- Bounding Box for Chennai ---
north, south, east, west = 13.14, 12.97, 80.29, 80.08  # Chennai region
graph = ox.graph_from_bbox(north, south, east, west, network_type="drive", retain_all=True)

# --- Validate Graph ---
if len(graph.edges) == 0:
    st.error("❌ No edges found in the graph. Try a different bounding box.")
else:
    st.success(f"✅ Graph loaded with {len(graph.edges)} edges.")

# --- Data Extraction and Display ---
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Display File Preview
    st.write("### 📊 Parking Data Preview")
    st.write(df.head())

    # Extract Important Information
    st.write("### 🚦 Extracted Parking Details")

    # Check for common columns
    if {'Latitude', 'Longitude', 'Action'}.issubset(df.columns):
        
        # Identify congested and available parking spots
        congested_df = df[df["Action"].isin(["Searching", "Left"])]
        available_df = df[df["Action"] == "Parked"]

        # Display statistics
        st.write(f"🔴 **Congested Spots:** {len(congested_df)}")
        st.write(f"🟢 **Available Spots:** {len(available_df)}")

    else:
        st.warning("⚠️ CSV file missing required columns: Latitude, Longitude, Action.")
        congested_df = pd.DataFrame()
        available_df = pd.DataFrame()
else:
    st.write("📂 Please upload a CSV file to analyze parking data.")
    congested_df = pd.DataFrame()
    available_df = pd.DataFrame()

# --- Route Generation Controls ---
st.sidebar.header("🔀 Route Generator")

# Choose the number of random routes
num_routes = st.sidebar.slider("Number of Routes", min_value=1, max_value=5, value=3)

# Base Map
m = folium.Map(location=[13.0827, 80.2707], zoom_start=12)
marker_cluster = MarkerCluster().add_to(m)

# --- Generate Random Routes ---
st.write("### 🔥 Random Routes with Parking Data")

for _ in range(num_routes):
    try:
        # Select random origin and destination nodes
        orig_node = random.choice(list(graph.nodes()))
        dest_node = random.choice(list(graph.nodes()))

        # Ensure the nodes are connected
        if nx.has_path(graph, orig_node, dest_node):
            # Compute the shortest path
            route = nx.shortest_path(graph, orig_node, dest_node, weight="length")

            # Extract coordinates
            route_coords = [(graph.nodes[node]['y'], graph.nodes[node]['x']) for node in route]

            # Plot the route
            folium.PolyLine(
                locations=route_coords,
                color=random.choice(['blue', 'green', 'purple', 'orange']),
                weight=5,
                opacity=0.8,
                tooltip="Random Route"
            ).add_to(m)

            # Add markers for origin and destination
            orig_lat, orig_lon = graph.nodes[orig_node]['y'], graph.nodes[orig_node]['x']
            dest_lat, dest_lon = graph.nodes[dest_node]['y'], graph.nodes[dest_node]['x']

            folium.Marker([orig_lat, orig_lon], 
                          popup="Start Point", 
                          icon=folium.Icon(color="green")).add_to(marker_cluster)
            
            folium.Marker([dest_lat, dest_lon], 
                          popup="Destination", 
                          icon=folium.Icon(color="red")).add_to(marker_cluster)

        else:
            st.warning("⚠️ No path found between random nodes.")
    except Exception as e:
        st.error(f"❌ Error: {e}")

# --- Add Parking Markers (from uploaded file) ---
if not congested_df.empty and not available_df.empty:
    # Add congested parking markers
    for _, row in congested_df.iterrows():
        folium.Marker(
            location=[row["Latitude"], row["Longitude"]],
            popup=f"🚦 Congested Parking Spot",
            icon=folium.Icon(color="red")
        ).add_to(marker_cluster)

    # Add available parking markers
    for _, row in available_df.iterrows():
        folium.Marker(
            location=[row["Latitude"], row["Longitude"]],
            popup=f"✅ Available Parking Spot",
            icon=folium.Icon(color="green")
        ).add_to(marker_cluster)

# --- Display the map ---
folium_static(m)

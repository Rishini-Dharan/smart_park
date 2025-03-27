import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image
import csv
import datetime
import os
import pandas as pd

# ===============================
# CSV Logging Functions
# ===============================
CSV_FILE = "smart_parking_log.csv"

def initialize_csv():
    """Creates/initializes the CSV file with headers"""
    if not os.path.exists(CSV_FILE) or os.stat(CSV_FILE).st_size == 0:
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Vehicle Type", "Count"])

def log_vehicle(vehicle_type, count):
    """Logs vehicle detection to CSV with timestamp"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, vehicle_type, count])

def show_csv_history():
    """Displays the CSV file history"""
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        if not df.empty:
            st.subheader("Detection History")
            st.dataframe(df)
            st.line_chart(df.groupby('Vehicle Type').sum()['Count'])
        else:
            st.info("No detection history found.")

# Initialize CSV file
initialize_csv()

# ===============================
# Load YOLOv8 Model
# ===============================
model = YOLO("yolov8n.pt")

# ===============================
# Streamlit UI Configuration
# ===============================
st.title("🚗 Real-Time Smart Parking Analytics")
st.write("Live vehicle detection and tracking with CSV logging")

with st.sidebar:
    st.header("Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
    vehicle_classes = st.multiselect(
        "Vehicle Classes to Detect",
        options=['car', 'truck', 'bus', 'motorcycle'],
        default=['car', 'truck', 'bus', 'motorcycle']
    )

# ===============================
# Session State Management
# ===============================
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'cap' not in st.session_state:
    st.session_state.cap = None
if 'track_history' not in st.session_state:
    st.session_state.track_history = {}
if 'vehicle_counts' not in st.session_state:
    st.session_state.vehicle_counts = {}

# ===============================
# Camera Control Functions
# ===============================
def start_camera_feed():
    """Initializes camera and detection system"""
    st.session_state.camera_active = True
    st.session_state.cap = cv2.VideoCapture(0)
    st.session_state.vehicle_counts = {vt: 0 for vt in vehicle_classes}
    st.session_state.track_history = {}

def stop_camera_feed():
    """Releases camera resources"""
    st.session_state.camera_active = False
    if st.session_state.cap:
        st.session_state.cap.release()
    cv2.destroyAllWindows()

# ===============================
# UI Controls
# ===============================
col1, col2 = st.columns(2)
with col1:
    if st.button("🎥 Start Real-time Detection") and not st.session_state.camera_active:
        start_camera_feed()
with col2:
    if st.button("⏹️ Stop Camera") and st.session_state.camera_active:
        stop_camera_feed()

# ===============================
# Real-Time Processing Loop
# ===============================
if st.session_state.camera_active and st.session_state.cap.isOpened():
    frame_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    while st.session_state.camera_active:
        success, frame = st.session_state.cap.read()
        if not success:
            st.error("Failed to capture video feed")
            stop_camera_feed()
            break
        
        # Perform object tracking
        results = model.track(
            frame,
            persist=True,
            conf=confidence_threshold,
            classes=[2, 3, 5, 7],  # COCO class IDs for vehicles
            verbose=False
        )
        
        # Process detections
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.int().cpu().tolist()
            
            for box, track_id, cls in zip(boxes, track_ids, clss):
                label = model.names[cls]
                if label not in vehicle_classes:
                    continue
                
                # Update vehicle counts
                if track_id not in st.session_state.track_history:
                    st.session_state.vehicle_counts[label] += 1
                    log_vehicle(label, 1)  # Log each new detection
                
                # Store tracking history
                st.session_state.track_history[track_id] = label
                
                # Draw bounding boxes
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} #{track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, caption="Live Camera Feed", use_column_width=True)
        
        # Update statistics
        stats_placeholder.markdown(f"""
        **Live Vehicle Counts:**
        - Total Vehicles: {sum(st.session_state.vehicle_counts.values())}
        - Cars: {st.session_state.vehicle_counts.get('car', 0)}
        - Trucks: {st.session_state.vehicle_counts.get('truck', 0)}
        - Buses: {st.session_state.vehicle_counts.get('bus', 0)}
        - Motorcycles: {st.session_state.vehicle_counts.get('motorcycle', 0)}
        """)

# ===============================
# Image Upload Handling
# ===============================
st.subheader("Image Upload Detection")
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("🔍 Detect Vehicles in Image"):
        results = model(np.array(image), conf=confidence_threshold, verbose=False)
        counts = {vt: 0 for vt in vehicle_classes}
        
        # Process detections
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                label = result.names[cls]
                if label in vehicle_classes:
                    counts[label] += 1
        
        # Log and display results
        for vt, count in counts.items():
            if count > 0:
                log_vehicle(vt, count)
        st.success(f"Detected vehicles: {counts}")

# ===============================
# Display Historical Data
# ===============================
show_csv_history()

# Cleanup on app exit
if st.session_state.cap:
    st.session_state.cap.release()

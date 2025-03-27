import streamlit as st
import cv2
import torch
import numpy as np
import pandas as pd
import time
from datetime import datetime
from ultralytics import YOLO
from PIL import Image

# Initialize YOLO model
model = YOLO("yolov8n.pt")

# Initialize session state
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'cap' not in st.session_state:
    st.session_state.cap = None
if 'vehicle_counts' not in st.session_state:
    st.session_state.vehicle_counts = {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0}
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = pd.DataFrame(columns=['Timestamp', 'Vehicle Type', 'Count'])

# Streamlit UI
st.title("🚗 Smart Parking Analytics System")
st.write("Real-time vehicle detection and analytics with CSV logging")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
    vehicle_classes = st.multiselect(
        "Select Vehicle Classes",
        options=['car', 'truck', 'bus', 'motorcycle'],
        default=['car', 'truck', 'bus', 'motorcycle']
    )
    csv_file = st.file_uploader("Upload existing CSV (optional)", type=["csv"])

# Camera control buttons
col1, col2 = st.columns(2)
with col1:
    start_camera = st.button("🎥 Start Real-time Detection")
with col2:
    stop_camera = st.button("⏹️ Stop Camera")

# Initialize CSV if uploaded
if csv_file is not None:
    st.session_state.detection_history = pd.read_csv(csv_file)

def update_csv(vehicle_type):
    """Update CSV with new detection entry"""
    new_entry = {
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'Vehicle Type': vehicle_type,
        'Count': 1
    }
    st.session_state.detection_history = pd.concat(
        [st.session_state.detection_history, pd.DataFrame([new_entry])],
        ignore_index=True
    )

# Camera processing
if start_camera:
    st.session_state.camera_active = True
    st.session_state.cap = cv2.VideoCapture(0)
    st.session_state.vehicle_counts = {vt: 0 for vt in vehicle_classes}

if stop_camera:
    st.session_state.camera_active = False
    if st.session_state.cap:
        st.session_state.cap.release()
    st.rerun()

# Real-time detection loop
if st.session_state.camera_active and st.session_state.cap.isOpened():
    frame_placeholder = st.empty()
    stats_placeholder = st.empty()
    chart_placeholder = st.empty()
    
    while st.session_state.camera_active:
        ret, frame = st.session_state.cap.read()
        if not ret:
            st.error("Failed to capture frame")
            st.session_state.camera_active = False
            break
        
        # Process frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.track(frame_rgb, conf=confidence_threshold, persist=True, verbose=False)
        
        current_counts = {vt: 0 for vt in vehicle_classes}
        
        # Draw bounding boxes and count vehicles
        if results[0].boxes.id is not None:
            for box, track_id in zip(results[0].boxes, results[0].boxes.id):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                label = results[0].names[int(box.cls[0])]
                
                if label in vehicle_classes:
                    color = (0, 255, 0)
                    cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame_rgb, 
                                f"{label.upper()} {conf:.2f} ID:{int(track_id)}",
                                (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.7, color, 2)
                    current_counts[label] += 1
        
        # Update counts and CSV
        for vt in vehicle_classes:
            if current_counts[vt] > st.session_state.vehicle_counts[vt]:
                update_csv(vt)
            st.session_state.vehicle_counts[vt] = current_counts[vt]
        
        # Display frame and stats
        frame_placeholder.image(frame_rgb, caption="Live Camera Feed", use_column_width=True)
        
        # Show current counts
        stats_placeholder.write("### Current Vehicle Counts")
        cols = st.columns(len(vehicle_classes))
        for idx, vt in enumerate(vehicle_classes):
            cols[idx].metric(f"{vt.upper()}", st.session_state.vehicle_counts[vt])
        
        # Show historical data
        chart_placeholder.line_chart(
            st.session_state.detection_history.groupby('Vehicle Type').count()['Count']
        )

# Image upload handling
uploaded_file = st.file_uploader("Upload Image for Detection", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("🔍 Detect Vehicles in Image"):
        image_np = np.array(image)
        results = model(image_np, conf=confidence_threshold, verbose=False)
        
        detected_image = image_np.copy()
        img_counts = {vt: 0 for vt in vehicle_classes}
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                label = result.names[int(box.cls[0])]
                
                if label in vehicle_classes:
                    color = (0, 255, 0)
                    cv2.rectangle(detected_image, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(detected_image, 
                               f"{label.upper()} {conf:.2f}",
                               (x1, y1 - 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 
                               1.0, color, 2)
                    img_counts[label] += 1
        
        st.image(detected_image, caption="Detected Vehicles", use_column_width=True)
        st.write("### Detected Vehicles Count:")
        for vt, count in img_counts.items():
            st.write(f"{vt.upper()}: {count}")

# CSV Download and Display
if not st.session_state.detection_history.empty:
    st.download_button(
        label="📥 Download Detection Data",
        data=st.session_state.detection_history.to_csv(index=False).encode('utf-8'),
        file_name=f"vehicle_detection_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime='text/csv'
    )
    st.write("### Detection History")
    st.dataframe(st.session_state.detection_history)

# Cleanup
if not st.session_state.camera_active and st.session_state.cap:
    st.session_state.cap.release()
    cv2.destroyAllWindows()
    st.session_state.cap = None

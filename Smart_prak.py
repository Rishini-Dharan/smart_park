import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load the pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")

# Streamlit UI
st.title("🚗 Smart Parking System - Real-time Vehicle Detection")
st.write("Real-time vehicle detection using webcam or uploaded images")

# Session state initialization
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'cap' not in st.session_state:
    st.session_state.cap = None

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
    vehicle_classes = st.multiselect(
        "Select Vehicle Classes",
        options=['car', 'truck', 'bus', 'motorcycle'],
        default=['car', 'truck', 'bus', 'motorcycle']
    )

# Camera control buttons
col1, col2 = st.columns(2)
with col1:
    start_camera = st.button("🎥 Start Real-time Detection")
with col2:
    stop_camera = st.button("⏹️ Stop Camera")

if start_camera:
    st.session_state.camera_active = True
    st.session_state.cap = cv2.VideoCapture(0)

if stop_camera:
    st.session_state.camera_active = False
    if st.session_state.cap:
        st.session_state.cap.release()
    st.rerun()

# Real-time camera processing
if st.session_state.camera_active and st.session_state.cap.isOpened():
    frame_placeholder = st.empty()
    
    while st.session_state.camera_active:
        ret, frame = st.session_state.cap.read()
        if not ret:
            st.error("Failed to capture frame")
            st.session_state.camera_active = False
            break
        
        # Convert frame to RGB and process
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform detection
        results = model(frame_rgb, conf=confidence_threshold, verbose=False)
        
        # Draw bounding boxes
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                label = result.names[int(box.cls[0])]
                
                if label in vehicle_classes:
                    color = (0, 255, 0)  # Green color for vehicles
                    cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame_rgb, 
                               f"{label.upper()} {conf:.2f}",
                               (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, color, 2)
        
        # Display processed frame
        frame_placeholder.image(frame_rgb, caption="Live Camera Feed", use_column_width=True)

# File upload handling
uploaded_file = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("🔍 Detect Vehicles in Uploaded Image"):
        image_np = np.array(image)
        results = model(image_np, conf=confidence_threshold, verbose=False)
        
        # Draw bounding boxes on uploaded image
        detected_image = image_np.copy()
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                label = result.names[int(box.cls[0])]
                
                if label in vehicle_classes:
                    color = (0, 255, 0)  # Green color for vehicles
                    cv2.rectangle(detected_image, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(detected_image, 
                               f"{label.upper()} {conf:.2f}",
                               (x1, y1 - 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 
                               1.0, color, 2)
        
        st.image(detected_image, caption="Detected Vehicles", use_column_width=True)

# Cleanup when camera is turned off
if not st.session_state.camera_active and st.session_state.cap:
    st.session_state.cap.release()
    cv2.destroyAllWindows()
    st.session_state.cap = None

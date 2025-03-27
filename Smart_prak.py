import streamlit as st
import cv2
import time
import csv
import datetime
import os
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Initialize session state FIRST
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'cap' not in st.session_state:
    st.session_state.cap = None

# Load model and initialize CSV after session state
model = YOLO("yolov8n.pt")
CSV_FILE = "smart_parking_log.csv"

# Initialize CSV (keep your existing implementation)
def initialize_csv():
    if not os.path.exists(CSV_FILE) or os.stat(CSV_FILE).st_size == 0:
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Event", "Date", "Time"])

initialize_csv()

# ===============================
# Main UI Components
# ===============================
st.title("🚗 Smart Parking System - Camera Feed")
st.write("Click the button below to open your webcam.")

# ===============================
# Camera Control Buttons
# ===============================
col1, col2 = st.columns(2)
with col1:
    if st.button("🎥 Start Camera"):
        st.session_state.camera_active = True
        st.session_state.cap = cv2.VideoCapture(0)
        log_event("Camera Started")

with col2:
    if st.button("⏹️ Stop Camera"):
        st.session_state.camera_active = False
        if st.session_state.cap is not None:
            st.session_state.cap.release()
        log_event("Camera Stopped")

# ===============================
# Camera Feed Section
# ===============================
def show_camera_feed():
    frame_placeholder = st.empty()
    while st.session_state.camera_active and st.session_state.cap.isOpened():
        ret, frame = st.session_state.cap.read()
        if not ret:
            st.error("Failed to capture frame")
            break
        
        # Vehicle detection and annotation
        results = model.predict(frame, conf=0.5, classes=[2, 3, 5, 7])
        annotated_frame = results[0].plot()  # Use YOLO's built-in plotting
        
        # Convert to RGB and display
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, caption="Live Camera Feed", use_column_width=True)
        time.sleep(0.1)

    if st.session_state.cap is not None:
        st.session_state.cap.release()
        cv2.destroyAllWindows()

# ===============================
# File Upload Section
# ===============================
st.subheader("Upload an Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # (Keep your existing image processing code here)

# ===============================
# Event History Section
# ===============================
st.subheader("Event History")
# (Keep your existing CSV display code here)

# ===============================
# Main Execution Flow
# ===============================
if st.session_state.camera_active:
    show_camera_feed()
else:
    st.write("Camera is currently inactive")

# Always show these components
st.write("Other system controls and information below...")

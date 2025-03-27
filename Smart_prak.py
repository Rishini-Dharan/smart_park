import streamlit as st
import cv2
import time
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image
import csv
import datetime
import os

# ===============================
# CSV Logging Functions
# ===============================
CSV_FILE = "smart_parking_log.csv"

def initialize_csv():
    """Creates the CSV file with headers if it doesn't exist or is empty."""
    if not os.path.exists(CSV_FILE) or os.stat(CSV_FILE).st_size == 0:
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Vehicle Type", "Count", "Date", "Time"])

def log_vehicle(vehicle_type, count):
    """Logs a vehicle type with its count and current date/time to CSV."""
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([vehicle_type, count, date_str, time_str])

def show_csv_history():
    """Displays the CSV file history if available."""
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "r", encoding="utf-8") as file:
            csv_reader = csv.reader(file)
            data = list(csv_reader)
        if len(data) > 1:
            st.subheader("Detection History")
            st.table(data[1:])
        else:
            st.info("No detection history found.")
    else:
        st.info("No detection history found.")

# Initialize CSV file
initialize_csv()

# ===============================
# Load YOLOv8 Model
# ===============================
model = YOLO("yolov8n.pt")  # Ensure this file is in your working directory or provide the correct path

# ===============================
# Streamlit UI Header & Sidebar
# ===============================
st.title("🚗 Smart Parking System - Real-time Vehicle Detection")
st.write("Detect vehicles in real time using your webcam or by uploading an image.")

with st.sidebar:
    st.header("Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
    vehicle_classes = st.multiselect(
        "Select Vehicle Classes",
        options=['car', 'truck', 'bus', 'motorcycle'],
        default=['car', 'truck', 'bus', 'motorcycle']
    )

# ===============================
# Session State Initialization
# ===============================
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'cap' not in st.session_state:
    st.session_state.cap = None
if 'live_counts' not in st.session_state:
    st.session_state.live_counts = {}

# ===============================
# Camera Control Buttons
# ===============================
col1, col2 = st.columns(2)
with col1:
    start_camera = st.button("🎥 Start Real-time Detection")
with col2:
    stop_camera = st.button("⏹️ Stop Camera")

# Start camera
if start_camera:
    st.session_state.camera_active = True
    # Use CAP_DSHOW for Windows if needed; remove for other OS
    st.session_state.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    st.session_state.live_counts = {}  # Reset counts at start

# Stop camera
if stop_camera:
    st.session_state.camera_active = False

# ===============================
# Function to Process Detection on an Image
# ===============================
def detect_vehicles_on_frame(frame_rgb, conf_thresh, classes):
    """Detect vehicles on a given RGB frame and return the processed frame and a count dictionary."""
    results = model(frame_rgb, conf=conf_thresh, verbose=False)
    counts = {}
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            label = result.names[int(box.cls[0])]
            if label in classes and conf > conf_thresh:
                # Draw bounding box and label on the frame
                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_rgb, f"{label.upper()} {conf:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                counts[label] = counts.get(label, 0) + 1
    return frame_rgb, counts

# ===============================
# Real-time Camera Detection
# ===============================
if st.session_state.camera_active and st.session_state.cap is not None and st.session_state.cap.isOpened():
    frame_placeholder = st.empty()
    count_placeholder = st.empty()
    
    st.write("Initializing camera, please wait...")
    time.sleep(2)  # Warm-up delay

    while st.session_state.camera_active:
        ret, frame = st.session_state.cap.read()
        if not ret:
            st.error("Failed to capture frame from camera.")
            st.session_state.camera_active = False
            break
        
        # Convert the captured frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect vehicles and get counts for this frame
        processed_frame, frame_counts = detect_vehicles_on_frame(frame_rgb, confidence_threshold, vehicle_classes)
        
        # Update cumulative live counts
        for label, count in frame_counts.items():
            st.session_state.live_counts[label] = st.session_state.live_counts.get(label, 0) + count
        
        # Calculate total vehicles (cumulative)
        total_count = sum(st.session_state.live_counts.values())
        
        # Update placeholders
        count_placeholder.markdown(
            f"**Live Vehicle Counts:** {st.session_state.live_counts}  \n"
            f"**Total Vehicles (Session):** {total_count}"
        )
        frame_placeholder.image(processed_frame, caption="Live Camera Feed", use_column_width=True)
        
        # Small delay for smoother streaming
        time.sleep(0.1)
    
    # When the camera is stopped, release the resource and log final counts
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        cv2.destroyAllWindows()
        st.session_state.cap = None

    # Log final counts to CSV (only if we had a proper session)
    if st.session_state.live_counts:
        for vehicle_type, count in st.session_state.live_counts.items():
            log_vehicle(vehicle_type, count)
        st.success("Logged live detection counts to CSV.")
        st.write("Final Session Counts:", st.session_state.live_counts)
    else:
        st.info("No vehicles detected during the session.")

# ===============================
# File Upload Detection
# ===============================
st.subheader("Image Upload Detection")
uploaded_file = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    if st.button("🔍 Detect Vehicles in Uploaded Image"):
        image_np = np.array(image)
        processed_img, upload_counts = detect_vehicles_on_frame(image_np.copy(), confidence_threshold, vehicle_classes)
        total_upload = sum(upload_counts.values())
        st.image(processed_img, caption="Detected Vehicles", use_column_width=True)
        st.write("Detection Counts:", upload_counts)
        st.write("Total Vehicles Detected:", total_upload)
        # Log to CSV
        for vehicle_type, count in upload_counts.items():
            log_vehicle(vehicle_type, count)
        st.success("Logged uploaded image detection counts to CSV.")

# ===============================
# Display CSV History
# ===============================
st.subheader("Detection History")
show_csv_history()

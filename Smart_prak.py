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

# CSV file configuration and logging functions
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

# Initialize the CSV file
initialize_csv()

# Load the pre-trained YOLOv8 model for vehicle detection
model = YOLO("yolov8n.pt")

# Streamlit UI Header
st.title("🚗 Smart Parking System - Vehicle Detection")
st.write("Detect vehicles using your live camera feed or by uploading an image. The total counts will be displayed and logged to a CSV file.")

# Function to detect vehicles and count them
def detect_vehicles(image):
    results = model(image)  # Run YOLO detection
    image_np = image.copy()
    counts = {}  # Dictionary to hold counts for each vehicle type
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            label = result.names[int(box.cls[0])]
            if conf > 0.5 and label in ["car", "truck", "motorcycle"]:
                # Draw bounding box and label
                cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(image_np, f"{label} ({conf:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                counts[label] = counts.get(label, 0) + 1
    return image_np, counts

# --- Section 1: Image Upload Detection ---
st.subheader("Image Upload Detection")
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file))
    st.image(image, caption="Uploaded Image", use_column_width=True)
    if st.button("🚗 Detect Vehicles in Uploaded Image"):
        detected_image, counts = detect_vehicles(image)
        st.image(detected_image, caption="Detected Vehicles", use_column_width=True)
        st.write("Detection Counts:", counts)
        # Log each vehicle type count to CSV
        for vehicle_type, count in counts.items():
            log_vehicle(vehicle_type, count)
        st.success("Logged detection counts to CSV.")

# --- Section 2: Live Camera Detection ---
st.subheader("Live Camera Detection")
if "live_feed" not in st.session_state:
    st.session_state.live_feed = False

col1, col2 = st.columns(2)
start_live = col1.button("🎥 Start Live Camera Detection")
stop_live = col2.button("⏹ Stop Live Camera Detection")

if start_live:
    st.session_state.live_feed = True
if stop_live:
    st.session_state.live_feed = False

frame_placeholder = st.empty()

# Variable to accumulate final counts from the live session
final_counts = {}

if st.session_state.live_feed:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 640)
    cap.set(4, 480)
    st.write("Initializing camera, please wait...")
    time.sleep(2)  # Allow camera to warm up

    while st.session_state.live_feed:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Failed to capture frame.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected_frame, counts = detect_vehicles(frame_rgb)
        # Update final_counts (summing up counts from each frame)
        for key, value in counts.items():
            final_counts[key] = final_counts.get(key, 0) + value

        # Create overlay text for current frame counts
        overlay_text = " | ".join([f"{k}: {v}" for k, v in counts.items()])
        cv2.putText(detected_frame, overlay_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        frame_placeholder.image(detected_frame, channels="RGB", use_column_width=True)
        # A small delay for smoother streaming
        time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()
    st.write("Final counts from live detection:", final_counts)
    # Log final counts to CSV
    for vehicle_type, count in final_counts.items():
        log_vehicle(vehicle_type, count)
    st.success("Logged live detection counts to CSV.")

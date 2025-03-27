import streamlit as st
import cv2
import csv
import os
import datetime
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import av
import threading

# Load the pre-trained YOLOv8 model for vehicle detection
model = YOLO("yolov8n.pt")

# File to store vehicle logs
CSV_FILE = "smart_parking_log.csv"

def initialize_csv():
    """Creates the CSV file with headers if not exists or if empty."""
    if not os.path.exists(CSV_FILE) or os.stat(CSV_FILE).st_size == 0:
        try:
            with open(CSV_FILE, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["Vehicle Type", "Vehicle Model", "Action", "Date", "Time"])
        except Exception as e:
            st.error(f"CSV initialization error: {e}")

def log_vehicle(vehicle_type, vehicle_model, action):
    """Logs vehicle entry/exit with time and date."""
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    try:
        with open(CSV_FILE, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([vehicle_type, vehicle_model, action, date, time])
        st.success(f"Logged: {vehicle_type} {vehicle_model} {action} at {time} on {date}")
    except PermissionError:
        st.error("Error: Permission denied while writing to CSV. Ensure the file is not open elsewhere.")

# Initialize CSV file for logging
initialize_csv()

st.title("🚗 Smart Parking System - Video Detection")
st.write("Detect and log vehicle entry/exit using YOLOv8 and OpenCV")

# Define entry and exit zones (adjust as per your camera's view)
ENTRY_ZONE_Y = 200  # vertical threshold for an entry event
EXIT_ZONE_Y = 500   # vertical threshold for an exit event

# Session state to store vehicle count and logs
if 'vehicle_count' not in st.session_state:
    st.session_state.vehicle_count = 0
if 'logs' not in st.session_state:
    st.session_state.logs = []

# Video capture setup
cap = None
stop_event = threading.Event()

def process_frame(frame):
    """Process each frame with YOLO detection and tracking."""
    img = frame.to_ndarray(format="bgr24")
    results = model(img, verbose=False)  # Disable verbose output
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            label = result.names[int(box.cls[0])]
            
            # Only consider vehicles with sufficient confidence
            if conf > 0.5 and label in ["car", "truck", "motorcycle", "bus"]:
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                prev_center = st.session_state.get(f"last_center_{label}", None)
                st.session_state[f"last_center_{label}"] = center_y
                
                # Entry detection: moving downward across the ENTRY_ZONE_Y
                if prev_center is not None and prev_center < ENTRY_ZONE_Y <= center_y:
                    log_vehicle(label, "Unknown Model", "Entry")
                    st.session_state.vehicle_count += 1
                # Exit detection: moving upward across the EXIT_ZONE_Y
                elif prev_center is not None and prev_center > EXIT_ZONE_Y >= center_y:
                    log_vehicle(label, "Unknown Model", "Exit")
                    st.session_state.vehicle_count = max(0, st.session_state.vehicle_count - 1)

                # Draw detection results on the frame
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{label} ({conf:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the vehicle count on the frame
    cv2.putText(img, f"Vehicles Inside: {st.session_state.vehicle_count}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Draw entry and exit zones
    cv2.line(img, (0, ENTRY_ZONE_Y), (img.shape[1], ENTRY_ZONE_Y), (255, 0, 0), 2)
    cv2.putText(img, "Entry Zone", (50, ENTRY_ZONE_Y - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.line(img, (0, EXIT_ZONE_Y), (img.shape[1], EXIT_ZONE_Y), (0, 0, 255), 2)
    cv2.putText(img, "Exit Zone", (50, EXIT_ZONE_Y - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return img

def video_capture():
    """Video capture thread function."""
    global cap
    cap = cv2.VideoCapture(0)  # Use 0 for default camera
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame from camera")
            break
            
        # Process frame with YOLO
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame, verbose=False)
        
        # Draw detections (simplified for mobile)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                label = result.names[int(box.cls[0])]
                
                if conf > 0.5 and label in ["car", "truck", "motorcycle", "bus"]:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the frame in Streamlit
        stframe.image(frame, channels="RGB", use_column_width=True)
        
    if cap:
        cap.release()

# Streamlit UI
col1, col2 = st.columns(2)
with col1:
    start_button = st.button("Start Detection")
with col2:
    stop_button = st.button("Stop Detection")

stframe = st.empty()

if start_button:
    stop_event.clear()
    thread = threading.Thread(target=video_capture)
    thread.start()

if stop_button:
    stop_event.set()
    if cap:
        cap.release()

# Display vehicle logs if available
if os.path.exists(CSV_FILE):
    with open(CSV_FILE, "r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        logs = list(csv_reader)
    
    if len(logs) > 1:
        st.subheader("📋 Vehicle Logs")
        st.table(logs[1:])  # Skip header row

# Mobile-friendly instructions
st.markdown("""
### 📱 Mobile Usage Instructions:
1. Open this page in Chrome or Safari
2. Grant camera permissions when prompted
3. Position your phone to view the parking area
4. Adjust the ENTRY_ZONE_Y and EXIT_ZONE_Y values in the code to match your camera view
""")

# Performance optimization tips
st.markdown("""
### ⚡ Performance Tips:
- Use good lighting conditions
- Position camera at a higher vantage point
- For better accuracy, train a custom YOLO model on parking lot images
""")

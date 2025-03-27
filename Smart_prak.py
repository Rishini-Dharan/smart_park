import streamlit as st
import cv2
import csv
import os
import datetime
import numpy as np
from ultralytics import YOLO

# Load the pre-trained YOLOv8 model for vehicle detection
model = YOLO("yolov8n.pt")

# File to store vehicle logs
CSV_FILE = "smart_parking_log.csv"

def initialize_csv():
    """Creates the CSV file with headers if not exists or if empty."""
    if not os.path.exists(CSV_FILE) or os.stat(CSV_FILE).st_size == 0:
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Vehicle Type", "Vehicle Model", "Action", "Date", "Time"])

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

# Initialize CSV
initialize_csv()

# Streamlit UI
st.title("🚗 Smart Parking System")
st.write("This system detects and logs vehicle entry/exit using YOLOv8 and OpenCV with a live camera feed.")

st.markdown(
    """
    **Note:** This app uses your webcam for live vehicle detection. Click 'Start Live Feed' to begin, and 'Stop Live Feed' to end the session.
    """
)

# Placeholder for the video feed
frame_placeholder = st.empty()

# Buttons to start and stop the live feed
if 'run' not in st.session_state:
    st.session_state.run = False

start_button = st.button("Start Live Feed")
stop_button = st.button("Stop Live Feed")

# Start or stop the live feed based on button clicks
if start_button:
    st.session_state.run = True

if stop_button:
    st.session_state.run = False

# Initialize OpenCV video capture
cap = None
if st.session_state.run:
    cap = cv2.VideoCapture(0)  # 0 for default webcam
    if not cap.isOpened():
        st.error("Error: Could not access the webcam. Please ensure it is connected and not in use by another application.")
        st.session_state.run = False

# Keep track of detected vehicles to avoid duplicate logging
detected_vehicles = set()

# Main loop for live feed
while st.session_state.run and cap is not None:
    ret, frame = cap.read()
    if not ret:
        st.error("Error: Could not read frame from webcam.")
        break

    # Run vehicle detection on the captured frame
    results = model(frame)

    # Process detections and draw bounding boxes/labels
    for result in results:
        for box in result.boxes:
            # Get box coordinates, confidence, and label
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            label = result.names[int(box.cls[0])]

            # Process only if confidence > 0.5 and for specific vehicle types
            if conf > 0.5 and label in ["car", "truck", "motorcycle"]:
                # Create a unique identifier for the vehicle (based on position and type)
                vehicle_id = f"{label}_{x1}_{y1}_{x2}_{y2}"

                # Log only if this vehicle hasn't been logged recently
                if vehicle_id not in detected_vehicles:
                    log_vehicle(label, "Unknown Model", "Entry")
                    detected_vehicles.add(vehicle_id)

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} ({conf:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

    # Clear old vehicle IDs to allow re-logging after some time (simulating vehicle exit/entry)
    if len(detected_vehicles) > 50:  # Arbitrary threshold to clear old detections
        detected_vehicles.clear()

    # Convert frame to RGB for display in Streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame, caption="Live Detection Results", channels="RGB")

# Release the capture when done
if cap is not None:
    cap.release()

# Display logs
if os.path.exists(CSV_FILE):
    with open(CSV_FILE, "r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        logs = list(csv_reader)
    
    if len(logs) > 1:
        st.subheader("📋 Vehicle Logs")
        st.table(logs[1:])  # Skip header row

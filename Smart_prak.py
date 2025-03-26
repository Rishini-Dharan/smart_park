import streamlit as st
import cv2
import torch
import csv
import os
import datetime
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# Load the pre-trained YOLOv8 model for vehicle detection
model = YOLO("yolov8n.pt")  # Using YOLOv8 nano model

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
st.write("This system detects and logs vehicle entry/exit using YOLOv8 and OpenCV.")

start_button = st.button("Start Detection")
stop_button = st.button("Stop Detection")

FRAME_WINDOW = st.empty()  # Placeholder for live video
LOG_WINDOW = st.empty()  # Placeholder for logs

vehicle_positions = defaultdict(lambda: None)  # Stores last known positions of detected vehicles
vehicle_count = 0  # Track number of vehicles inside

# Define entry and exit zones (Modify based on camera setup)
ENTRY_ZONE = (100, 200, 300, 400)  # Example coordinates
EXIT_ZONE = (400, 500, 600, 700)

# Open camera feed
if start_button:
    cap = cv2.VideoCapture(0)  # Change to video file path if needed

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)  # Detect vehicles in the frame

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                label = result.names[int(box.cls[0])]

                if conf > 0.5 and label in ["car", "truck", "motorcycle"]:
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    prev_position = vehicle_positions[label]
                    vehicle_positions[label] = center_y

                    if prev_position is not None:
                        if prev_position < ENTRY_ZONE[1] and center_y >= ENTRY_ZONE[1]:
                            log_vehicle(label, "Unknown Model", "Entry")
                            vehicle_count += 1
                        elif prev_position > EXIT_ZONE[1] and center_y <= EXIT_ZONE[1]:
                            log_vehicle(label, "Unknown Model", "Exit")
                            vehicle_count -= 1

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(frame, f"Vehicles Inside: {vehicle_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Convert frame to RGB for Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame, channels="RGB")

        if stop_button:
            break

    cap.release()
    cv2.destroyAllWindows()

# Show vehicle log table
if os.path.exists(CSV_FILE):
    with open(CSV_FILE, "r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        logs = list(csv_reader)

    if len(logs) > 1:
        st.subheader("📋 Vehicle Logs")
        LOG_WINDOW.table(logs[1:])  # Skip header row

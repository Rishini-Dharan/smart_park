import streamlit as st
import cv2
import os
import csv
import datetime
from collections import defaultdict
from ultralytics import YOLO

# Constants
CSV_FILE = "smart_parking_log.csv"
ENTRY_ZONE = (100, 200, 300, 400)  # Example entry zone coordinates
EXIT_ZONE = (400, 500, 600, 700)  # Example exit zone coordinates

# Initialize YOLO model
model = YOLO("yolov8n.pt")

def initialize_csv():
    """Creates the CSV file with headers if not exists or if empty."""
    if not os.path.exists(CSV_FILE) or os.stat(CSV_FILE).st_size == 0:
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Vehicle Type", "Vehicle Model", "Action", "Date", "Time"])

def log_vehicle(vehicle_type, action):
    """Logs vehicle entry/exit with time and date."""
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    try:
        with open(CSV_FILE, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([vehicle_type, "Unknown Model", action, date, time])
        st.success(f"Logged: {vehicle_type} {action} at {time} on {date}")
    except PermissionError:
        st.error("Error: Permission denied while writing to CSV. Ensure the file is not open elsewhere.")

def process_frame(frame, vehicle_positions, vehicle_count):
    """Detect vehicles and handle entry/exit logic."""
    results = model(frame)
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
                        log_vehicle(label, "Entry")
                        vehicle_count += 1
                    elif prev_position > EXIT_ZONE[1] and center_y <= EXIT_ZONE[1]:
                        log_vehicle(label, "Exit")
                        vehicle_count -= 1

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, vehicle_positions, vehicle_count

# Initialize CSV file
initialize_csv()

# Streamlit UI
st.title("🚗 Smart Parking System")
st.write("This system detects and logs vehicle entry/exit using YOLOv8 and OpenCV.")

# Session state variables
if 'run_detection' not in st.session_state:
    st.session_state.run_detection = False
if 'cap' not in st.session_state:
    st.session_state.cap = None
if 'vehicle_positions' not in st.session_state:
    st.session_state.vehicle_positions = defaultdict(lambda: None)
if 'vehicle_count' not in st.session_state:
    st.session_state.vehicle_count = 0

# UI for control buttons
start_button = st.button("Start Detection")
stop_button = st.button("Stop Detection")
FRAME_WINDOW = st.empty()  # Placeholder for live video

if start_button:
    st.session_state.run_detection = True
    if st.session_state.cap is None:
        st.session_state.cap = cv2.VideoCapture(0)

if stop_button:
    st.session_state.run_detection = False
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None

if st.session_state.run_detection and st.session_state.cap is not None:
    while st.session_state.run_detection:
        ret, frame = st.session_state.cap.read()
        if ret:
            frame, st.session_state.vehicle_positions, st.session_state.vehicle_count = process_frame(
                frame, st.session_state.vehicle_positions, st.session_state.vehicle_count
            )

            # Display vehicle count
            cv2.putText(frame, f"Vehicles Inside: {st.session_state.vehicle_count}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Convert frame to RGB and display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame, channels="RGB")
        else:
            st.error("Failed to capture frame. Stopping detection.")
            st.session_state.run_detection = False
            st.session_state.cap.release()
            st.session_state.cap = None
            break

# Display logs
if os.path.exists(CSV_FILE):
    with open(CSV_FILE, "r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        logs = list(csv_reader)

    if len(logs) > 1:
        st.subheader("📋 Vehicle Logs")
        st.table(logs[1:])  # Skip header row

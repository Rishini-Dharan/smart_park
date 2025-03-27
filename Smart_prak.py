import streamlit as st
import cv2
import csv
import os
import datetime
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# Load the pre-trained YOLOv8 model for vehicle detection
# Ensure that the model file "yolov8n.pt" is in your working directory or provide the correct path.
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
        st.info(f"Logged: {vehicle_type} {vehicle_model} {action} at {time} on {date}")
    except PermissionError:
        st.error("Error: Permission denied while writing to CSV. Ensure the file is not open elsewhere.")

# Initialize CSV file for logging
initialize_csv()

st.title("🚗 Smart Parking System - Video Detection")
st.write("This system detects and logs vehicle entry/exit using YOLOv8 and OpenCV via continuous video streaming.")

# Define entry and exit zones (example coordinates; adjust for your setup)
ENTRY_ZONE_Y = 200  # vertical threshold for an entry event
EXIT_ZONE_Y = 500   # vertical threshold for an exit event

# For simplicity, we are tracking the last known vertical center per vehicle type.
# For multiple vehicles, consider using object trackers with unique IDs.
class YOLOVideoTransformer(VideoTransformerBase):
    def __init__(self):
        # Tracks the last center_y per vehicle label.
        self.last_center = {}
        self.vehicle_count = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img)
        
        # Process each detection result
        for result in results:
            # Loop over detected boxes
            for box in result.boxes:
                # Get box coordinates and confidence
                coords = box.xyxy[0]
                x1, y1, x2, y2 = map(int, coords)
                conf = box.conf[0].item()
                label = result.names[int(box.cls[0])]

                # Filter only relevant vehicles
                if conf > 0.5 and label in ["car", "truck", "motorcycle"]:
                    # Compute the center of the bounding box
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    prev_center = self.last_center.get(label, None)
                    self.last_center[label] = center_y  # update the last center for this label

                    # Simple entry/exit logic based on vertical movement
                    if prev_center is not None:
                        # Detect an entry event (moving downward crossing ENTRY_ZONE_Y)
                        if prev_center < ENTRY_ZONE_Y and center_y >= ENTRY_ZONE_Y:
                            log_vehicle(label, "Unknown Model", "Entry")
                            self.vehicle_count += 1
                        # Detect an exit event (moving upward crossing EXIT_ZONE_Y)
                        elif prev_center > EXIT_ZONE_Y and center_y <= EXIT_ZONE_Y:
                            log_vehicle(label, "Unknown Model", "Exit")
                            self.vehicle_count = max(0, self.vehicle_count - 1)

                    # Draw the bounding box and label on the frame
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"{label} ({conf:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display vehicle count on the frame
        cv2.putText(img, f"Vehicles Inside: {self.vehicle_count}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        return img

# RTCConfiguration (using default ICE servers)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Start the video streamer with the YOLO transformer
webrtc_streamer(
    key="smart-parking-video",
    video_transformer_factory=YOLOVideoTransformer,
    rtc_configuration=RTC_CONFIGURATION,
)

# Display the vehicle logs (if available)
if os.path.exists(CSV_FILE):
    with open(CSV_FILE, "r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        logs = list(csv_reader)
    
    if len(logs) > 1:
        st.subheader("📋 Vehicle Logs")
        st.table(logs[1:])  # Exclude the header row

import streamlit as st
import cv2
import csv
import os
import datetime
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

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
        st.info(f"Logged: {vehicle_type} {vehicle_model} {action} at {time} on {date}")
    except PermissionError:
        st.error("Error: Permission denied while writing to CSV. Ensure the file is not open elsewhere.")

# Initialize CSV file for logging
initialize_csv()

st.title("🚗 Smart Parking System - Video Detection")
st.write("Detect and log vehicle entry/exit using YOLOv8 and OpenCV via continuous video streaming.")

# Define entry and exit zones (adjust as per your camera's view)
ENTRY_ZONE_Y = 200  # vertical threshold for an entry event
EXIT_ZONE_Y = 500   # vertical threshold for an exit event

# Video Transformer for processing each video frame
class YOLOVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_center = {}
        self.vehicle_count = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img)
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                label = result.names[int(box.cls[0])]
                
                # Only consider vehicles with sufficient confidence
                if conf > 0.5 and label in ["car", "truck", "motorcycle"]:
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    prev_center = self.last_center.get(label, None)
                    self.last_center[label] = center_y  # Update the last seen vertical position
                    
                    # Entry detection: moving downward across the ENTRY_ZONE_Y
                    if prev_center is not None and prev_center < ENTRY_ZONE_Y <= center_y:
                        log_vehicle(label, "Unknown Model", "Entry")
                        self.vehicle_count += 1
                    # Exit detection: moving upward across the EXIT_ZONE_Y
                    elif prev_center is not None and prev_center > EXIT_ZONE_Y >= center_y:
                        log_vehicle(label, "Unknown Model", "Exit")
                        self.vehicle_count = max(0, self.vehicle_count - 1)

                    # Draw detection results on the frame
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"{label} ({conf:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the vehicle count on the frame
        cv2.putText(img, f"Vehicles Inside: {self.vehicle_count}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        return img

# RTCConfiguration with default ICE servers
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Start the WebRTC video streamer with the YOLO transformer
webrtc_streamer(
    key="smart-parking-video",
    video_transformer_factory=YOLOVideoTransformer,
    rtc_configuration=RTC_CONFIGURATION,
)

# Display vehicle logs if available
if os.path.exists(CSV_FILE):
    with open(CSV_FILE, "r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        logs = list(csv_reader)
    
    if len(logs) > 1:
        st.subheader("📋 Vehicle Logs")
        st.table(logs[1:])  # Skip header row

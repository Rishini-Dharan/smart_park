import streamlit as st
import cv2
import csv
import os
import datetime
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import threading
import queue

# Load the pre-trained YOLOv8 model for vehicle detection
model = YOLO("yolov8n.pt")

# File to store vehicle logs
CSV_FILE = "smart_parking_log.csv"

# Configuration
ENTRY_ZONE_Y = 200
EXIT_ZONE_Y = 500
FRAME_QUEUE = queue.Queue(maxsize=1)
STOP_EVENT = threading.Event()

def initialize_csv():
    """Initialize CSV log file."""
    if not os.path.exists(CSV_FILE) or os.stat(CSV_FILE).st_size == 0:
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Vehicle Type", "Vehicle Model", "Action", "Date", "Time"])

def log_vehicle(vehicle_type, action):
    """Log vehicle entry/exit with timestamp."""
    now = datetime.datetime.now()
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            vehicle_type,
            "Unknown Model",
            action,
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S")
        ])

def video_processing():
    """Main video processing loop."""
    cap = cv2.VideoCapture(0)
    track_history = defaultdict(lambda: [])
    
    try:
        while not STOP_EVENT.is_set():
            success, frame = cap.read()
            if not success:
                st.error("Failed to capture video")
                break
            
            results = model.track(frame, persist=True, verbose=False)
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                clss = results[0].boxes.cls.cpu().tolist()
                
                for box, track_id, cls in zip(boxes, track_ids, clss):
                    label = model.names[int(cls)]
                    if label not in ["car", "truck", "motorcycle"]:
                        continue
                    
                    x1, y1, x2, y2 = map(int, box)
                    center_y = (y1 + y2) // 2
                    
                    # Track movement history
                    track = track_history[track_id]
                    track.append(center_y)
                    if len(track) > 30:  # Keep last 30 positions
                        track.pop(0)
                    
                    # Detect direction
                    if len(track) >= 2:
                        direction = track[-1] - track[-2]
                        
                        # Entry detection
                        if center_y > ENTRY_ZONE_Y and direction > 0:
                            log_vehicle(label, "Entry")
                        
                        # Exit detection
                        if center_y < EXIT_ZONE_Y and direction < 0:
                            log_vehicle(label, "Exit")
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} #{track_id}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Put processed frame in queue
            if FRAME_QUEUE.empty():
                FRAME_QUEUE.put(frame)
                
    finally:
        cap.release()
        STOP_EVENT.clear()

# Streamlit UI
st.title("🚗 Smart Parking System")
initialize_csv()

# Start/Stop controls
col1, col2 = st.columns(2)
with col1:
    if st.button("Start Detection", key="start"):
        STOP_EVENT.clear()
        threading.Thread(target=video_processing, daemon=True).start()

with col2:
    if st.button("Stop Detection", key="stop"):
        STOP_EVENT.set()

# Display video feed
video_placeholder = st.empty()

while not STOP_EVENT.is_set():
    try:
        frame = FRAME_QUEUE.get(timeout=1)
        video_placeholder.image(frame, channels="BGR", use_column_width=True)
    except queue.Empty:
        continue

# Display logs
if os.path.exists(CSV_FILE):
    with open(CSV_FILE, "r", encoding="utf-8") as file:
        st.subheader("Vehicle Logs")
        st.dataframe(pd.read_csv(file))

st.markdown("""
### 📱 Mobile Tips:
1. Use Chrome or Safari
2. Rotate phone to landscape
3. Position camera 2-3 meters above ground
4. Ensure good lighting
""")

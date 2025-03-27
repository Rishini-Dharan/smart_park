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
    except Exception as e:
        st.error(f"An error occurred while logging: {e}")

# Initialize CSV
initialize_csv()

# Streamlit UI
st.title("🚗 Smart Parking System")
st.write("This system detects and logs vehicle entry/exit using YOLOv8 and OpenCV.")

st.markdown(
    """
    **Note:** In this deployment, please use the browser camera to capture an image for detection.
    """
)

# Use Streamlit's camera input widget to capture an image from the user's browser
image_file = st.camera_input("Take a picture for detection")

if image_file is not None:
    # Convert the uploaded image file to a format OpenCV can work with
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    try:
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
                    # For demonstration, we log every detection as an "Entry"
                    log_vehicle(label, "Unknown Model", "Entry")

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

    except Exception as e:
        st.error(f"Error during detection: {e}")

    # Convert frame to RGB for display in Streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame, caption="Detection Results", channels="RGB")
else:
    st.info("Please capture an image using your camera for detection.")

# Display logs
if os.path.exists(CSV_FILE):
    with open(CSV_FILE, "r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        logs = list(csv_reader)

    # Skip the header row and display the rest
    if len(logs) > 1:
        st.subheader("📋 Vehicle Logs")
        st.table(logs[1:])  # Correctly skip the header row

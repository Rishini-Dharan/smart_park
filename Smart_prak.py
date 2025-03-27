import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
import csv
import os
import datetime
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

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
st.write("This system detects and logs vehicle entry/exit using YOLOv8.")

# Initialize session state variables
if 'vehicle_positions' not in st.session_state:
    st.session_state.vehicle_positions = defaultdict(lambda: None)
if 'vehicle_count' not in st.session_state:
    st.session_state.vehicle_count = 0

# Define entry and exit zones (Modify based on camera setup)
ENTRY_ZONE = (100, 200, 300, 400)  # Example coordinates (x1, y1, x2, y2)
EXIT_ZONE = (400, 500, 600, 700)

# Upload image
image_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if image_file is not None:
    # Read the image using Pillow
    image = Image.open(image_file)
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)

    # Convert image to numpy array for YOLO processing
    frame = np.array(image)

    # Vehicle detection and processing
    results = model(frame)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            label = result.names[int(box.cls[0])]

            if conf > 0.5 and label in ["car", "truck", "motorcycle"]:
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                prev_position = st.session_state.vehicle_positions[label]
                st.session_state.vehicle_positions[label] = center_y

                if prev_position is not None:
                    # Entry/Exit logic
                    if prev_position < ENTRY_ZONE[1] and center_y >= ENTRY_ZONE[1]:
                        log_vehicle(label, "Unknown Model", "Entry")
                        st.session_state.vehicle_count += 1
                    elif prev_position > EXIT_ZONE[1] and center_y <= EXIT_ZONE[1]:
                        log_vehicle(label, "Unknown Model", "Exit")
                        st.session_state.vehicle_count -= 1

                # Draw bounding boxes and labels
                draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
                font = ImageFont.load_default()
                draw.text((x1, y1 - 10), f"{label} ({conf:.2f})", font=font, fill="green")

    # Display vehicle count
    font = ImageFont.load_default()
    draw.text((50, 50), f"Vehicles Inside: {st.session_state.vehicle_count}", font=font, fill="yellow")

    # Display the processed image
    st.image(image, caption="Processed Image with Vehicle Detection", use_column_width=True)

# Display logs
if os.path.exists(CSV_FILE):
    with open(CSV_FILE, "r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        logs = list(csv_reader)
    if len(logs) > 1:
        st.subheader("📋 Vehicle Logs")
        st.table(logs[1:])  # Skip header row

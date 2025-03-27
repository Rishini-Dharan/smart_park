import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load the pre-trained YOLOv8 model for vehicle detection
model = YOLO("yolov8n.pt")

# Streamlit UI
st.title("🚗 Smart Parking System - Vehicle Detection")
st.write("Capture an image and detect vehicles in it using YOLOv8.")

# Initialize session state for camera
if 'cap' not in st.session_state:
    st.session_state.cap = None

# Capture image button
capture_button = st.button("Capture Image")

if capture_button:
    st.session_state.cap = cv2.VideoCapture(0)  # Open webcam
    ret, frame = st.session_state.cap.read()
    if ret:
        # Convert the captured frame to RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.session_state.image = frame  # Store the image in session state
        st.image(frame, caption="Captured Image", use_column_width=True)
    else:
        st.error("Failed to capture image.")
    
    # Release the camera
    st.session_state.cap.release()
    st.session_state.cap = None

# Detect Vehicles Button
if 'image' in st.session_state and st.button("Detect Vehicles"):
    image = st.session_state.image

    # Convert the image to the format required by YOLO
    image_np = np.array(image)

    # Perform vehicle detection
    results = model(image_np)

    # Draw bounding boxes and labels
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            label = result.names[int(box.cls[0])]

            if conf > 0.5 and label in ["car", "truck", "motorcycle"]:
                cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(image_np, f"{label} ({conf:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the image with detected vehicles
    st.image(image_np, caption="Detected Vehicles", use_column_width=True)

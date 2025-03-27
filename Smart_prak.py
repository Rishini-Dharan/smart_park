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
st.write("Capture an image or upload an image and detect vehicles in it using YOLOv8.")

# Camera Capture Button
capture_button = st.button("📷 Capture Image from Camera")

# File Upload Option
uploaded_file = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])

# Initialize image variable
image = None

# Capture image from webcam
if capture_button:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use CAP_DSHOW to prevent errors on Windows
    cap.set(3, 640)  # Set width
    cap.set(4, 480)  # Set height

    # Wait for camera to initialize
    st.write("Initializing camera, please wait...")
    for _ in range(5):  # Wait for a few frames to stabilize
        ret, frame = cap.read()

    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = frame
        st.image(image, caption="Captured Image", use_column_width=True)
    else:
        st.error("❌ Failed to capture image. Please try again.")

    # Release camera
    cap.release()
    cv2.destroyAllWindows()

# If an image is uploaded, use it instead
if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file))
    st.image(image, caption="Uploaded Image", use_column_width=True)

# Detect Vehicles Button
if image is not None and st.button("🚗 Detect Vehicles"):
    # Perform vehicle detection
    results = model(image)

    # Convert image to OpenCV format
    image_np = image.copy()

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

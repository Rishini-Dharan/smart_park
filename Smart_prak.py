import streamlit as st
import cv2
import time
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load the pre-trained YOLOv8 model for vehicle detection
model = YOLO("yolov8n.pt")

# Streamlit UI
st.title("🚗 Smart Parking System - Vehicle Detection")
st.write("Detect vehicles using real-time camera feed or upload an image for detection.")

# File Upload Option
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

# Camera Live Stream Button
start_camera = st.button("🎥 Start Live Camera Detection")

# Function to perform vehicle detection
def detect_vehicles(image):
    results = model(image)  # Perform YOLO detection
    image_np = image.copy()  # Copy image for modifications

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

    return image_np

# Process uploaded image
if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file))
    detected_image = detect_vehicles(image)
    st.image(detected_image, caption="🚗 Detected Vehicles", use_column_width=True)

# Real-Time Camera Detection
if start_camera:
    st.write("📸 **Starting Camera... Press 'Q' to stop.**")
    cap = cv2.VideoCapture(0)  # Open webcam

    if not cap.isOpened():
        st.error("❌ Camera not available. Please check your settings.")
    else:
        # Loop for live camera feed
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to capture frame.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame color
            detected_frame = detect_vehicles(frame)  # Detect vehicles

            # Display frame in Streamlit
            st.image(detected_frame, channels="RGB", use_column_width=True)

            # Press 'Q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()  # Release camera
        cv2.destroyAllWindows()  # Close OpenCV windows

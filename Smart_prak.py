import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import av

# Load YOLO model
try:
    model = YOLO("yolov8n.pt")
except Exception as e:
    st.error(f"Error loading YOLO model: {e}")
    st.stop()

# Define video processor for live feed
class VehicleDetector(VideoProcessorBase):
    def __init__(self):
        self.model = model

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.model(img)
        for result in results:
            for box in result.boxes:
                if box.conf > 0.5 and result.names[int(box.cls)] in ["car", "truck", "motorcycle"]:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"{result.names[int(box.cls)]} {box.conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit app
st.title("Vehicle Detection System")

option = st.selectbox("Choose input method", ["Live Camera Feed", "Upload Image"])

if option == "Live Camera Feed":
    st.write("Allow camera access when prompted.")
    webrtc_streamer(key="example", video_processor_factory=VehicleDetector)
elif option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            image_rgb = np.array(image)
            st.image(image_rgb, caption="Uploaded Image", use_column_width=True)
            if st.button("Detect Vehicles"):
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                results = model(image_bgr)
                for result in results:
                    for box in result.boxes:
                        if box.conf > 0.5 and result.names[int(box.cls)] in ["car", "truck", "motorcycle"]:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(image_bgr, f"{result.names[int(box.cls)]} {box.conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                image_rgb_detected = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                st.image(image_rgb_detected, caption="Detected Vehicles", use_column_width=True)
        except Exception as e:
            st.error(f"Error processing image: {e}")

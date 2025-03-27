import streamlit as st
import cv2
import time
import csv
import datetime
import os
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Initialize session state variables FIRST
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'cap' not in st.session_state:
    st.session_state.cap = None

# Then load other components
model = YOLO("yolov8n.pt")
CSV_FILE = "smart_parking_log.csv"

# Rest of your code remains the same...
# ===============================
# CSV Logging Functions
# ===============================
def initialize_csv():
    """Creates the CSV file with headers if it doesn't exist or is empty."""
    if not os.path.exists(CSV_FILE) or os.stat(CSV_FILE).st_size == 0:
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Event", "Date", "Time"])

# ... keep all other functions and logic the same ...

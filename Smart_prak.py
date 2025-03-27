import streamlit as st
import cv2
import time
import csv
import datetime
import os

# ===============================
# CSV Logging Functions
# ===============================
CSV_FILE = "smart_parking_log.csv"

def initialize_csv():
    """Creates the CSV file with headers if it doesn't exist or is empty."""
    if not os.path.exists(CSV_FILE) or os.stat(CSV_FILE).st_size == 0:
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Event", "Date", "Time"])

def log_event(event):
    """Logs an event (e.g., 'Camera Started') with timestamp."""
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([event, date_str, time_str])

def show_csv_history():
    """Displays the CSV file history if available."""
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "r", encoding="utf-8") as file:
            csv_reader = csv.reader(file)
            data = list(csv_reader)
        if len(data) > 1:
            st.subheader("Event History")
            st.table(data[1:])
        else:
            st.info("No event history found.")
    else:
        st.info("No event history found.")

# Initialize CSV file
initialize_csv()

# ===============================
# Streamlit UI Header & Sidebar
# ===============================
st.title("🚗 Webcam Viewer")
st.write("Click the button below to open your webcam.")

# ===============================
# Session State Initialization
# ===============================
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'cap' not in st.session_state:
    st.session_state.cap = None

# ===============================
# Camera Control Buttons
# ===============================
col1, col2 = st.columns(2)
with col1:
    start_camera = st.button("🎥 Start Camera")
with col2:
    stop_camera = st.button("⏹️ Stop Camera")

# Start camera
if start_camera:
    st.session_state.camera_active = True
    st.session_state.cap = cv2.VideoCapture(0)  # Open camera
    if st.session_state.cap.isOpened():
        log_event("Camera Started")  # Log event
    else:
        st.error("Failed to open the camera. Please check your settings.")
        st.session_state.camera_active = False

# Stop camera
if stop_camera:
    st.session_state.camera_active = False
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        cv2.destroyAllWindows()
        st.session_state.cap = None
    log_event("Camera Stopped")  # Log event

# ===============================
# Real-time Camera Streaming
# ===============================
if st.session_state.camera_active and st.session_state.cap is not None and st.session_state.cap.isOpened():
    frame_placeholder = st.empty()
    st.write("**Live Camera Feed is running...**")
    
    while st.session_state.camera_active:
        ret, frame = st.session_state.cap.read()
        if not ret:
            st.error("Failed to capture frame from the camera.")
            st.session_state.camera_active = False
            break
        
        # Convert frame to RGB for display in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, caption="Live Camera Feed", use_column_width=True)
        
        # Small delay for smoother streaming
        time.sleep(0.1)

    # Release camera when stopped
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        cv2.destroyAllWindows()
        st.session_state.cap = None

# ===============================
# Display CSV History
# ===============================
st.subheader("Event History")
show_csv_history()

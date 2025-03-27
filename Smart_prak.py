# ... (keep all previous imports and initializations)

# ===============================
# Real-time Camera Streaming with Vehicle Detection
# ===============================
if st.session_state.camera_active and st.session_state.cap is not None and st.session_state.cap.isOpened():
    frame_placeholder = st.empty()
    while st.session_state.camera_active:
        ret, frame = st.session_state.cap.read()
        if not ret:
            st.error("Failed to capture frame from camera.")
            st.session_state.camera_active = False
            break
        
        # Perform vehicle detection
        results = model.predict(frame, conf=0.5, classes=[2, 3, 5, 7])  # Filter for vehicles only
        
        # Process detections and draw bounding boxes
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                label = result.names[int(box.cls[0])]
                
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
        
        # Convert frame to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, caption="Live Camera Feed with Vehicle Detection", use_column_width=True)
        
        # Small delay for smoother streaming
        time.sleep(0.1)

# ... (keep the rest of the code unchanged)

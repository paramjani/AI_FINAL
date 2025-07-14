import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import os
import time
import pandas as pd
from datetime import datetime

# Load YOLOv8 model (make sure your model is in the project directory)
MODEL_PATH = "best.pt"  # Change this to your trained model path
model = YOLO(MODEL_PATH)

# Define the classes you're interested in detecting
CLASS_NAMES = model.names  # Auto-fetch class names from YOLO

# CSV log file
LOG_FILE = "violation_logs.csv"

# Streamlit setup
st.set_page_config(page_title="AI CCTV Surveillance", layout="wide")
st.title("🎯 AI-Powered CCTV Surveillance System for Jyoti CNC")
st.markdown("Detects anomalies, safety breaches, and human presence using live CCTV or video.")

# Sidebar controls
source_type = st.sidebar.radio("Select Input Source", ['Webcam', 'Upload Video', 'Upload Image'])


# Temporary file location
temp_dir = tempfile.mkdtemp()

# Violation logger
def log_violation(class_name, confidence):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = pd.DataFrame([[timestamp, class_name, round(confidence, 2)]], columns=["Timestamp", "Violation", "Confidence"])
    
    if os.path.exists(LOG_FILE):
        entry.to_csv(LOG_FILE, mode='a', header=False, index=False)
    else:
        entry.to_csv(LOG_FILE, mode='w', header=True, index=False)

# Frame processor with logging
def process_frame(frame):
    results = model(frame)[0]
    annotated_frame = results.plot()

    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = CLASS_NAMES[cls_id]
        confidence = float(box.conf[0])

        # Log violations that start with "NO"
        if "NO" in class_name.upper():
            log_violation(class_name, confidence)

    return annotated_frame, results

# Video uploader and player
def display_video(video_path):
    cap = cv2.VideoCapture(video_path)
    st_frame = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, _ = process_frame(frame)
        st_frame.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
        time.sleep(0.03)

    cap.release()

# Main logic
if source_type == 'Upload Video':
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    if uploaded_file:
        temp_video_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_video_path, 'wb') as f:
            f.write(uploaded_file.read())

        st.success("Video uploaded. Processing...")
        display_video(temp_video_path)

elif source_type == 'Webcam':
    run_webcam = st.button("Start Webcam")

    if run_webcam:
        cap = cv2.VideoCapture(0)
        st_frame = st.empty()

        st.info("Press 'Stop' to end webcam stream.")
        stop = st.button("Stop")

        while cap.isOpened() and not stop:
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame, _ = process_frame(frame)
            st_frame.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

        cap.release()

# Display violation logs
st.markdown("## 📄 Violation Logs")
if os.path.exists(LOG_FILE):
    df_logs = pd.read_csv(LOG_FILE)
    st.dataframe(df_logs.tail(10))  # Show last 10 logs
    st.download_button("📥 Download Full Log", data=df_logs.to_csv(index=False), file_name="violation_logs.csv", mime="text/csv")
else:
    st.info("No violations logged yet.")

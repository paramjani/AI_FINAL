import streamlit as st
import cv2
import tempfile
import os
import time
from ultralytics import YOLO
from datetime import datetime
import pandas as pd

# Initialize
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names
LOG_FILE = "violation_logs.csv"

# Streamlit settings
st.set_page_config(page_title="📱 Mobile Camera PPE Detection", layout="wide")
st.title("📱 AI PPE Detection from Android IP Camera (YOLOv8)")
st.markdown("This app performs real-time safety gear (PPE) detection from your Android device via RTSP stream.")

# Sidebar - Input RTSP stream
rtsp_url = st.sidebar.text_input(
    "📡 RTSP Stream URL (from your mobile IP Webcam app)",
    placeholder="rtsp://192.168.29.129:8080/h264_ulaw.sdp"
)

def log_violation(class_name, confidence):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = pd.DataFrame([[timestamp, class_name, round(confidence, 2)]], columns=["Timestamp", "Violation", "Confidence"])
    if os.path.exists(LOG_FILE):
        entry.to_csv(LOG_FILE, mode='a', header=False, index=False)
    else:
        entry.to_csv(LOG_FILE, mode='w', header=True, index=False)

def process_frame(frame):
    results = model(frame)[0]
    annotated = results.plot()
    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = CLASS_NAMES[cls_id]
        confidence = float(box.conf[0])
        if "NO" in class_name.upper():
            log_violation(class_name, confidence)
    return annotated

def run_rtsp_stream(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)
    st_frame = st.empty()
    stop_btn = st.button("🛑 Stop Detection")

    if not cap.isOpened():
        st.error("❌ Failed to open stream. Check IP, port, and Wi-Fi.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Lost connection to camera.")
            break

        annotated = process_frame(frame)
        st_frame.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

        if stop_btn:
            break

        time.sleep(0.03)

    cap.release()

if rtsp_url:
    if st.button("▶️ Start Detection"):
        run_rtsp_stream(rtsp_url)
else:
    st.info("Enter your IP Webcam RTSP URL in the sidebar to start.")

# Logs
st.markdown("## 📝 Violation Log")
if os.path.exists(LOG_FILE):
    df = pd.read_csv(LOG_FILE)
    st.dataframe(df.tail(10))
    st.download_button("📥 Download Log", df.to_csv(index=False), "violation_logs.csv", "text/csv")
else:
    st.info("No violations logged yet.")

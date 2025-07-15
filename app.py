import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import os
import time
import pandas as pd
from datetime import datetime

# Load YOLOv8 model
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names

LOG_FILE = "violation_logs.csv"
st.set_page_config(page_title="AI CCTV Surveillance", layout="wide")
st.title("🎯 AI-Powered CCTV Surveillance System")
st.markdown("**Developed for Jyoti CNC | Detects PPE violations, anomalies & human presence from video/image inputs.**")

# Sidebar: Branding and controls
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f1/Jyoti_CNC_Automation_Limited_Logo.png/1200px-Jyoti_CNC_Automation_Limited_Logo.png", width=150)
    st.markdown("### 🎛️ System Controls")
    source_type = st.radio("Select Input Type", ['Upload Video', 'Upload Image'])
    voice_alert = st.toggle("🔊 Enable Voice Alerts", value=True)
    st.markdown("---")
    st.markdown("**Created by Param Jani**  \nFinal Year Computer Engineering  \n[GTU Robotics Club]")

# Temp folder
temp_dir = tempfile.mkdtemp()

# Log violation
def log_violation(class_name, confidence):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = pd.DataFrame([[timestamp, class_name, round(confidence, 2)]], columns=["Timestamp", "Violation", "Confidence"])
    if os.path.exists(LOG_FILE):
        entry.to_csv(LOG_FILE, mode='a', header=False, index=False)
    else:
        entry.to_csv(LOG_FILE, mode='w', header=True, index=False)
    if voice_alert:
        engine.say(f"Violation detected: {class_name}")
        engine.runAndWait()

# Process frame
def process_frame(frame):
    results = model(frame)[0]
    annotated_frame = results.plot()
    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = CLASS_NAMES[cls_id]
        confidence = float(box.conf[0])
        if "NO" in class_name.upper():
            log_violation(class_name, confidence)
    return annotated_frame, results

# Stream processor
def display_video(video_source):
    cap = cv2.VideoCapture(video_source)
    st_frame = st.empty()
    stop_button = st.button("🛑 Stop Stream")
    fail_count = 0

    while cap.isOpened():
        if stop_button:
            break
        ret, frame = cap.read()
        if not ret:
            fail_count += 1
            if fail_count > 10:
                st.warning("⚠️ Stream lost or ended.")
                break
            continue
        fail_count = 0
        annotated_frame, _ = process_frame(frame)
        st_frame.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
        time.sleep(0.03)
    cap.release()

# Image processor
def process_image(image_path):
    frame = cv2.imread(image_path)
    annotated_frame, _ = process_frame(frame)
    return annotated_frame

# Main logic
st.markdown("### 📥 Upload Input File")

if source_type == 'Upload Video':
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file:
        temp_video_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_video_path, 'wb') as f:
            f.write(uploaded_file.read())
        st.video(uploaded_file)
        st.success("✅ Video Uploaded. Starting Analysis...")
        display_video(temp_video_path)

elif source_type == 'Upload Image':
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        temp_image_path = os.path.join(temp_dir, uploaded_image.name)
        with open(temp_image_path, 'wb') as f:
            f.write(uploaded_image.read())
        st.image(uploaded_image, caption="📸 Uploaded Image", use_column_width=True)
        st.success("✅ Image Uploaded. Processing...")
        annotated_image = process_image(temp_image_path)
        st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="Processed Image", use_column_width=True)

# Logs section
st.markdown("## 📄 Violation Logs")
if os.path.exists(LOG_FILE):
    df_logs = pd.read_csv(LOG_FILE)
    st.metric("🚨 Total Violations", len(df_logs))
    today = datetime.now().date()
    today_count = df_logs[pd.to_datetime(df_logs['Timestamp']).dt.date == today].shape[0]
    st.metric("📅 Today's Violations", today_count)

    st.dataframe(df_logs.tail(10))
    st.download_button("📥 Download Full Log", data=df_logs.to_csv(index=False), file_name="violation_logs.csv", mime="text/csv")

    if st.button("🗑️ Clear Logs (Admin)"):
        os.remove(LOG_FILE)
        st.experimental_rerun()
else:
    st.info("✅ No violations detected yet.")

# Footer
st.markdown("---")
st.markdown("""
> 💼 **Note:** This system is designed for industrial safety compliance and anomaly detection.  
> 💡 Need help? Contact: [param.jani@gtu.edu](mailto:param.jani@gtu.edu)
""")

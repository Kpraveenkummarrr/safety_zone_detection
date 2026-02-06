import streamlit as st
import cv2
import numpy as np
import time
import os
from datetime import datetime
from ultralytics import YOLO

# ===================== CONFIG =====================
st.set_page_config(
    page_title="Safety Zone Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_CONF_DEFAULT = 0.4
ALERT_DELAY = 2  # seconds inside zone before violation

# ===================== UI HEADER =====================
st.title("🚧 AI-Based Safety Zone Detection System")
st.caption("Real-time restricted area monitoring with smart alerts")

# ===================== LOAD MODEL =====================
model = YOLO("yolov8n.pt")

# ===================== SESSION STATE =====================
if "violations" not in st.session_state:
    st.session_state.violations = 0
if "inside_since" not in st.session_state:
    st.session_state.inside_since = {}
if "snapshots" not in st.session_state:
    st.session_state.snapshots = 0

# ===================== SIDEBAR =====================
st.sidebar.header("⚙️ Control Panel")

run = st.sidebar.toggle("▶ Start Camera", value=False)

st.sidebar.subheader("🔲 Restricted Area (Rectangle)")
zone_x = st.sidebar.slider("X Position", 0, 500, 150)
zone_y = st.sidebar.slider("Y Position", 0, 400, 100)
zone_w = st.sidebar.slider("Width", 50, 500, 300)
zone_h = st.sidebar.slider("Height", 50, 400, 250)

confidence = st.sidebar.slider(
    "🎯 Detection Confidence",
    0.2, 0.9, MODEL_CONF_DEFAULT
)

show_zone = st.sidebar.checkbox("Show Restricted Zone", True)
save_snapshot = st.sidebar.checkbox("📸 Save Snapshot on Violation", True)
enable_timer = st.sidebar.checkbox("⏱ Enable Delay Timer", True)

st.sidebar.markdown("---")

if st.sidebar.button("🗑 Reset Violation Count"):
    st.session_state.violations = 0
    st.session_state.snapshots = 0

st.sidebar.metric("🚨 Violations", st.session_state.violations)
st.sidebar.metric("📸 Snapshots", st.session_state.snapshots)

# ===================== STATUS PANEL =====================
col1, col2, col3 = st.columns(3)
status_box = col1.empty()
time_box = col2.empty()
fps_box = col3.empty()

# ===================== CAMERA =====================
frame_window = st.image([], channels="RGB")
cap = None

if run:
    cap = cv2.VideoCapture(0)

prev_time = time.time()

# ===================== MAIN LOOP =====================
while run and cap and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("❌ Camera not available")
        break

    h, w, _ = frame.shape

    # Scale rectangle to camera resolution
    rx1 = int(zone_x * w / 600)
    ry1 = int(zone_y * h / 400)
    rx2 = int((zone_x + zone_w) * w / 600)
    ry2 = int((zone_y + zone_h) * h / 400)

    if show_zone:
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
        cv2.putText(frame, "RESTRICTED ZONE",
                    (rx1, ry1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)

    results = model(frame, conf=confidence)[0]
    violation_now = False

    for box in results.boxes:
        cls = int(box.cls[0])
        if model.names[cls] == "person":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2)//2, (y1 + y2)//2

            inside = rx1 <= cx <= rx2 and ry1 <= cy <= ry2
            pid = f"{cx}_{cy}"

            if inside:
                if pid not in st.session_state.inside_since:
                    st.session_state.inside_since[pid] = time.time()

                duration = time.time() - st.session_state.inside_since[pid]

                if not enable_timer or duration >= ALERT_DELAY:
                    violation_now = True
                    st.session_state.violations += 1

                    if save_snapshot:
                        os.makedirs("violations", exist_ok=True)
                        fname = f"violations/violation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        cv2.imwrite(fname, frame)
                        st.session_state.snapshots += 1

                    color = (0, 0, 255)
                    label = "⚠ RESTRICTED AREA"
                else:
                    color = (0, 255, 255)
                    label = "INSIDE ZONE"
            else:
                st.session_state.inside_since.pop(pid, None)
                color = (0, 255, 0)
                label = "SAFE"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)

    # ===================== STATUS =====================
    if violation_now:
        status_box.error("🚨 SAFETY VIOLATION DETECTED")
    else:
        status_box.success("✅ AREA SAFE")

    time_box.info(f"⏱ Alert Delay: {ALERT_DELAY}s")

    # FPS
    now = time.time()
    fps = int(1 / (now - prev_time))
    prev_time = now
    fps_box.info(f"🎞 FPS: {fps}")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_window.image(frame)

# ===================== CLEANUP =====================
if cap:
    cap.release()

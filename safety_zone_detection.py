import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Safety Zone Detection", layout="wide")
st.title("🚧 Safety Zone Detection System")

model = YOLO("yolov8n.pt")

st.sidebar.header("Controls")
run = st.sidebar.checkbox("Start Camera")
st.sidebar.info("Draw polygon to mark restricted area")

# Canvas
canvas = st_canvas(
    fill_color="rgba(255,0,0,0.3)",
    stroke_width=3,
    stroke_color="#FF0000",
    background_color="#000000",
    height=400,
    width=600,
    drawing_mode="polygon",
    key="canvas",
)

# ✅ CORRECT polygon extraction
zone_points = []

if canvas.json_data is not None:
    for obj in canvas.json_data["objects"]:
        if obj["type"] == "path":
            for item in obj["path"]:
                if item[0] in ["M", "L"]:   # Move / Line
                    x = int(item[1])
                    y = int(item[2])
                    zone_points.append((x, y))

frame_window = st.image([])

cap = None
if run:
    cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Camera not detected")
        break

    # Draw safety zone
    if len(zone_points) >= 3:
        cv2.polylines(
            frame,
            [np.array(zone_points, np.int32)],
            True,
            (0, 255, 255),
            2
        )

    result = model(frame)[0]

    for box in result.boxes:
        cls = int(box.cls[0])
        if model.names[cls] == "person":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2)//2, (y1 + y2)//2

            status = "SAFE"
            color = (0, 255, 0)

            if len(zone_points) >= 3:
                inside = cv2.pointPolygonTest(
                    np.array(zone_points, np.int32),
                    (cx, cy),
                    False
                )
                if inside >= 0:
                    status = "ALERT: Restricted Zone"
                    color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, status, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_window.image(frame)

if cap:
    cap.release()

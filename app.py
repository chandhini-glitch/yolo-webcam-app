import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

st.title("📷 Real-Time Object Detection with YOLOv8")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

run = st.checkbox("Turn on Webcam")
FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(0)

    st.markdown("📸 Press `Stop` to turn off webcam")

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Webcam not detected!")
            break

        results = model(frame)
        annotated_frame = results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        FRAME_WINDOW.image(annotated_frame)

    cap.release()
else:
    st.write("👆 Turn on the checkbox to activate webcam.")

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
import cv2
import numpy as np

st.title("📷 Real-Time Object Detection with YOLOv8")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

class YOLOTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = model(img)
        annotated = results[0].plot()

        return annotated

# Stream webcam using WebRTC
webrtc_streamer(key="yolo", video_transformer_factory=YOLOTransformer)


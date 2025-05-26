import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from ultralytics import YOLO
import cv2
import numpy as np

st.set_page_config(page_title="YOLOv8 Live Detection", layout="centered")
st.title("📷 Real-Time Object Detection with YOLOv8")

# Load YOLOv8 model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Fix WebRTC connection with public STUN server
rtc_config = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Define transformation logic
class YOLOv8Transformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img)
        annotated = results[0].plot()
        return annotated

# Start webcam stream
webrtc_streamer(
    key="yolov8-stream",
    video_transformer_factory=YOLOv8Transformer,
    rtc_configuration=rtc_config
)



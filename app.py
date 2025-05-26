import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from ultralytics import YOLO
import cv2
import numpy as np

st.set_page_config(page_title="YOLOv8 Live Detection", layout="centered")
st.title("📷 Real-Time Object Detection with YOLOv8")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # Make sure model file is downloaded automatically

model = load_model()

rtc_config = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class YOLOv8Transformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img)
        annotated_frame = results[0].plot()
        return annotated_frame

webrtc_streamer(
    key="yolov8-stream",
    video_transformer_factory=YOLOv8Transformer,
    rtc_configuration=rtc_config,
    media_stream_constraints={"video": True, "audio": False},
)




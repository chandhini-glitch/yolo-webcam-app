import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from ultralytics import YOLO
import cv2
import numpy as np

st.set_page_config(page_title="YOLOv8 Live Detection", layout="centered")
st.title("📷 Real-Time Object Detection with YOLOv8")

st.info("Click DONE after selecting webcam input to start detection.")

@st.cache_resource
def load_model():
    try:
        model = YOLO("yolov8n.pt")  # Use Nano version (smallest)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None

model = load_model()

rtc_config = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class YOLOv8Transformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if model is not None:
            try:
                results = model(img)
                annotated_frame = results[0].plot()
                return annotated_frame
            except Exception as e:
                st.error(f"Detection error: {e}")
                return img
        else:
            return img

webrtc_streamer(
    key="yolov8-stream",
    video_transformer_factory=YOLOv8Transformer,
    rtc_configuration=rtc_config,
    media_stream_constraints={"video": True, "audio": False},
)



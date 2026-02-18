import streamlit as st
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from PIL import Image

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="AI Face Mask Detection", layout="wide")

# -------------------------
# Detect Cloud Environment
# -------------------------
IS_CLOUD = os.getenv("STREAMLIT_SHARING_MODE") == "true"

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("📦 Project Info")
st.sidebar.markdown("""
**Model:** MobileNetV2  
**Framework:** TensorFlow  
**Deployment:** Streamlit Cloud  
**Developer:** Shivam 🚀  
""")

if IS_CLOUD:
    mode = st.sidebar.radio("Choose Mode", ["Upload Image"])
    st.sidebar.warning("⚠ Live webcam disabled on Streamlit Cloud")
else:
    mode = st.sidebar.radio("Choose Mode", ["Upload Image", "Live Webcam"])

theme_toggle = st.sidebar.toggle("🌙 Dark Mode", value=True)

# -------------------------
# Dynamic Theme
# -------------------------
if theme_toggle:
    bg = "#0f172a"
    text = "white"
else:
    bg = "white"
    text = "black"

st.markdown(f"""
<style>
body {{
    background-color: {bg};
    color: {text};
}}

.conf-box {{
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 22px;
}}

.success {{
    background-color: rgba(16,185,129,0.2);
    border: 1px solid #10b981;
}}

.error {{
    background-color: rgba(239,68,68,0.2);
    border: 1px solid #ef4444;
}}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Load Model (Cached)
# -------------------------
@st.cache_resource
def load_trained_model():
    return load_model("face_mask_model.keras")

model = load_trained_model()
IMG_SIZE = 224

# -------------------------
# Face Detector
# -------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -------------------------
# IMAGE MODE
# -------------------------
if mode == "Upload Image":

    st.title("😷 AI Face Mask Detection")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = img_np[y:y+h, x:x+w]
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            face = face / 255.0
            face = np.expand_dims(face, axis=0)

            prediction = model.predict(face, verbose=0)[0][0]

            label = "Mask" if prediction < 0.5 else "No Mask"
            color = (0,255,0) if prediction < 0.5 else (255,0,0)

            cv2.rectangle(img_np, (x,y), (x+w,y+h), color, 3)
            cv2.putText(img_np, label, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        st.image(img_np, caption="Processed Image", use_container_width=True)

# -------------------------
# WEBCAM MODE (Local Only)
# -------------------------
if not IS_CLOUD and mode == "Live Webcam":

    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

    class VideoProcessor(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x,y,w,h) in faces:
                face = img[y:y+h, x:x+w]
                face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                face = face / 255.0
                face = np.expand_dims(face, axis=0)

                prediction = model.predict(face, verbose=0)[0][0]

                label = "Mask" if prediction < 0.5 else "No Mask"
                color = (0,255,0) if prediction < 0.5 else (0,0,255)

                cv2.rectangle(img, (x,y), (x+w,y+h), color, 3)
                cv2.putText(img, label, (x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            return img

    st.title("🎥 Live Mask Detection (Local Only)")
    webrtc_streamer(key="mask-detection", video_processor_factory=VideoProcessor)

import streamlit as st
import numpy as np
import cv2
import os
import mediapipe as mp   
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
**Face Detection:** MediaPipe  
**Deployment:** Streamlit Cloud  
**Developer:** Shivam 🚀  
""")

mode = st.sidebar.radio("Choose Mode", ["Upload Image"])

theme_toggle = st.sidebar.toggle("🌙 Dark Mode", value=True)

# -------------------------
# Theme
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
</style>
""", unsafe_allow_html=True)

# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_trained_model():
    return load_model("face_mask_model.keras")

model = load_trained_model()
IMG_SIZE = 224

# -------------------------
# MediaPipe Setup
# -------------------------
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)

# -------------------------
# IMAGE MODE
# -------------------------
st.title("😷 AI Face Mask Detection")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    results = face_detection.process(img_np)

    if results.detections:

        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box

            h, w, _ = img_np.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            # Crop face
            face = img_np[y:y+height, x:x+width]

            if face.size == 0:
                continue

            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            face = face / 255.0
            face = np.expand_dims(face, axis=0)

            prediction = model.predict(face, verbose=0)[0][0]

            label = "Mask" if prediction < 0.5 else "No Mask"
            color = (0,255,0) if prediction < 0.5 else (255,0,0)

            # Draw box
            cv2.rectangle(img_np, (x,y), (x+width,y+height), color, 3)
            cv2.putText(img_np, label, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        st.success("Face detected successfully!")
        st.image(img_np, caption="Processed Image", width="stretch")

    else:
        st.warning("⚠ No face detected in the image.")
        st.image(img_np, caption="Uploaded Image", width="stretch")

import streamlit as st
import numpy as np
import cv2
import os
import mediapipe as mp
from tensorflow.keras.models import load_model
from PIL import Image

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Face Mask Detection",
    page_icon="😷",
    layout="wide"
)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.title("📦 Project Info")
st.sidebar.markdown("""
**Model:** MobileNetV2  
**Framework:** TensorFlow  
**Face Detection:** MediaPipe  
**Deployment:** Streamlit Cloud  
**Developer:** Shivam 🚀  
""")

mode = st.sidebar.radio("Choose Mode", ["Upload Image"])
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=True)

# ---------------------------------------------------
# THEME
# ---------------------------------------------------
if dark_mode:
    background = "#0f172a"
    text_color = "white"
else:
    background = "white"
    text_color = "black"

st.markdown(f"""
    <style>
        body {{
            background-color: {background};
            color: {text_color};
        }}
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODEL (CACHED)
# ---------------------------------------------------
@st.cache_resource
def load_trained_model():
    return load_model("face_mask_model.keras")

model = load_trained_model()
IMG_SIZE = 224

# ---------------------------------------------------
# MEDIAPIPE FACE DETECTOR (CACHED)
# ---------------------------------------------------
@st.cache_resource
def load_face_detector():
    mp_face = mp.solutions.face_detection
    return mp_face.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.3  # Reduced for masked faces
    )

face_detector = load_face_detector()

# ---------------------------------------------------
# MAIN TITLE
# ---------------------------------------------------
st.title("😷 AI Face Mask Detection")
st.markdown("Upload an image to detect whether a person is wearing a mask.")

# ---------------------------------------------------
# IMAGE UPLOAD
# ---------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    # Load Image
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    # Convert to RGB for MediaPipe
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    # Detect Faces
    results = face_detector.process(img_rgb)

    if results.detections:

        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box

            h, w, _ = img_np.shape
            x = max(0, int(bbox.xmin * w))
            y = max(0, int(bbox.ymin * h))
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            face = img_np[y:y+height, x:x+width]

            if face.size == 0:
                continue

            # Preprocess for model
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            face = face / 255.0
            face = np.expand_dims(face, axis=0)

            prediction = model.predict(face, verbose=0)[0][0]

            label = "Mask" if prediction < 0.5 else "No Mask"
            confidence = round(float(1 - prediction if prediction < 0.5 else prediction) * 100, 2)

            color = (0, 255, 0) if label == "Mask" else (255, 0, 0)

            # Draw Bounding Box
            cv2.rectangle(img_np, (x, y), (x + width, y + height), color, 3)

            # Put Label
            cv2.putText(
                img_np,
                f"{label} ({confidence}%)",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2
            )

        st.success("✅ Face detected successfully!")
        st.image(img_np, caption="Processed Image", use_container_width=True)

    else:
        st.warning("⚠ No face detected in the image.")
        st.image(img_np, caption="Uploaded Image", use_container_width=True)

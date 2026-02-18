import streamlit as st
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

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
# LOAD MODEL
# ---------------------------------------------------
@st.cache_resource
def load_trained_model():
    return load_model("face_mask_model.keras")

model = load_trained_model()
IMG_SIZE = 224

# ---------------------------------------------------
# LOAD MEDIAPIPE
# ---------------------------------------------------
@st.cache_resource
def load_face_detector():
    mp_face = mp.solutions.face_detection
    return mp_face.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.3
    )

face_detector = load_face_detector()

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
st.title("😷 AI Face Mask Detection")
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    results = face_detector.process(img_np)

    if results.detections:

        draw = ImageDraw.Draw(image)

        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box

            w, h = image.size
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            face = img_np[y:y+height, x:x+width]

            if face.size == 0:
                continue

            # Resize using PIL instead of cv2
            face_pil = Image.fromarray(face).resize((IMG_SIZE, IMG_SIZE))
            face_array = np.array(face_pil) / 255.0
            face_array = np.expand_dims(face_array, axis=0)

            prediction = model.predict(face_array, verbose=0)[0][0]

            label = "Mask" if prediction < 0.5 else "No Mask"
            confidence = round(float(1 - prediction if prediction < 0.5 else prediction) * 100, 2)

            color = "green" if label == "Mask" else "red"

            # Draw rectangle
            draw.rectangle(
                [x, y, x + width, y + height],
                outline=color,
                width=4
            )

            # Draw label
            draw.text(
                (x, y - 20),
                f"{label} ({confidence}%)",
                fill=color
            )

        st.success("✅ Face detected successfully!")
        st.image(image, use_container_width=True)

    else:
        st.warning("⚠ No face detected.")
        st.image(image, use_container_width=True)

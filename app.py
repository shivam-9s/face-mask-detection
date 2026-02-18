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
# Sidebar
# -------------------------
st.sidebar.title("📦 Project Info")
st.sidebar.markdown("""
**Model:** MobileNetV2  
**Framework:** TensorFlow  
**Face Detection:** Haar Cascade  
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
# Haar Cascade Setup
# -------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -------------------------
# IMAGE MODE
# -------------------------
st.title("😷 AI Face Mask Detection")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        st.warning("⚠ No face detected in the image.")
        st.image(img_np, caption="Uploaded Image", width="stretch")

    else:
        for (x, y, w, h) in faces:

            face = img_np[y:y+h, x:x+w]

            if face.size == 0:
                continue

            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            face = face / 255.0
            face = np.expand_dims(face, axis=0)

            prediction = model.predict(face, verbose=0)[0][0]

            label = "Mask" if prediction < 0.5 else "No Mask"
            color = (0,255,0) if prediction < 0.5 else (255,0,0)

            cv2.rectangle(img_np, (x,y), (x+w,y+h), color, 3)
            cv2.putText(img_np, label, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        st.success("Face detected successfully!")
        st.image(img_np, caption="Processed Image", width="stretch")

import streamlit as st
import numpy as np
import cv2
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
**Deployment:** Streamlit Cloud  
**Developer:** Shivam 🚀  
""")

st.sidebar.success("✅ Live Webcam disabled for Cloud stability")

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
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    font-size: 20px;
    margin-top: 15px;
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
# MAIN UI
# -------------------------
st.title("😷 AI Face Mask Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.warning("⚠ No face detected in the image.")
        st.image(img_np, caption="Uploaded Image", width="stretch")
    else:
        for (x, y, w, h) in faces:
            face = img_np[y:y+h, x:x+w]
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            face = face / 255.0
            face = np.expand_dims(face, axis=0)

            prediction = model.predict(face, verbose=0)[0][0]

            if prediction < 0.5:
                label = "Mask"
                confidence = (1 - prediction) * 100
                color = (0, 255, 0)
                box_class = "success"
            else:
                label = "No Mask"
                confidence = prediction * 100
                color = (255, 0, 0)
                box_class = "error"

            cv2.rectangle(img_np, (x, y), (x+w, y+h), color, 3)
            cv2.putText(img_np,
                        f"{label} ({confidence:.2f}%)",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2)

        st.image(img_np, caption="Processed Image", width="stretch")

        st.markdown(
            f"<div class='conf-box {box_class}'>Prediction: {label} ({confidence:.2f}% confidence)</div>",
            unsafe_allow_html=True
        )

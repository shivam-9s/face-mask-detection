import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="AI Face Mask Detector",
    page_icon="😷",
    layout="centered"
)

# -----------------------------
# Custom Premium CSS
# -----------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

body {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #1f1c2c);
    background-size: 400% 400%;
    animation: gradientBG 12s ease infinite;
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

.main {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    padding: 25px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
}

h1 {
    text-align: center;
    font-size: 42px;
    font-weight: 600;
    background: linear-gradient(90deg, #00f2fe, #4facfe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.stFileUploader {
    background: rgba(255,255,255,0.1);
    padding: 15px;
    border-radius: 15px;
    border: 1px solid rgba(255,255,255,0.2);
}

.pred-card {
    padding: 20px;
    border-radius: 15px;
    font-size: 22px;
    text-align: center;
    font-weight: 500;
    margin-top: 20px;
    transition: 0.3s ease-in-out;
}

.success {
    background: rgba(16,185,129,0.2);
    color: #10b981;
    border: 1px solid #10b981;
}

.error {
    background: rgba(239,68,68,0.2);
    color: #ef4444;
    border: 1px solid #ef4444;
}

footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model (Cached)
# -----------------------------
@st.cache_resource
def load_trained_model():
    return load_model("face_mask_model.keras")

model = load_trained_model()

IMG_SIZE = 224

# -----------------------------
# UI
# -----------------------------
st.title("🤖 AI Face Mask Detection")

st.markdown(
    "<p style='text-align:center; color: #ccc;'>Upload an image to detect whether a person is wearing a mask.</p>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    img = np.array(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

    with st.spinner("🧠 AI is analyzing the image..."):
        prediction = model.predict(img)[0][0]

    confidence = round((1 - prediction) * 100, 2) if prediction < 0.5 else round(prediction * 100, 2)

    if prediction > 0.5:
        st.markdown(f"""
        <div class="pred-card error">
        ❌ No Mask Detected <br>
        Confidence: {confidence}%
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="pred-card success">
        ✅ Mask Detected <br>
        Confidence: {confidence}%
        </div>
        """, unsafe_allow_html=True)

    st.progress(int(confidence))

st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; font-size:14px; color:#aaa;'>Built with ❤️ using MobileNetV2 & Streamlit</p>",
    unsafe_allow_html=True
)

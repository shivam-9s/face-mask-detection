import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="centered"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}

.main {
    background: linear-gradient(135deg, #1f2937, #111827);
    border-radius: 15px;
    padding: 20px;
}

h1 {
    text-align: center;
    color: #00f2fe;
    font-weight: 700;
}

.stFileUploader {
    background-color: #1f2937;
    padding: 15px;
    border-radius: 10px;
}

.pred-box {
    padding: 15px;
    border-radius: 12px;
    font-size: 20px;
    text-align: center;
    font-weight: bold;
}

.success-box {
    background-color: #065f46;
    color: white;
}

.error-box {
    background-color: #7f1d1d;
    color: white;
}
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
st.title("😷 Face Mask Detection App")
st.write("Upload an image to check whether the person is wearing a mask.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    img = np.array(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

    with st.spinner("Analyzing Image..."):
        prediction = model.predict(img)[0][0]

    st.subheader("Prediction Result")

    if prediction > 0.5:
        confidence = round(prediction * 100, 2)
        st.markdown(f"""
        <div class="pred-box error-box">
        ❌ No Mask Detected<br>
        Confidence: {confidence}%
        </div>
        """, unsafe_allow_html=True)
    else:
        confidence = round((1 - prediction) * 100, 2)
        st.markdown(f"""
        <div class="pred-box success-box">
        ✅ Mask Detected<br>
        Confidence: {confidence}%
        </div>
        """, unsafe_allow_html=True)

    st.progress(int(confidence))

st.markdown("<hr>", unsafe_allow_html=True)
st.caption("🚀 Built with MobileNetV2 | Streamlit | Deep Learning")

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

    # Preprocess Image
    img = np.array(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

    with st.spinner("Analyzing Image..."):
        prediction = model.predict(img)[0][0]

    st.subheader("Prediction Result:")

    if prediction > 0.5:
        confidence = round(prediction * 100, 2)
        st.error("❌ No Mask Detected")
        st.progress(int(confidence))
        st.write(f"Confidence: {confidence}%")
    else:
        confidence = round((1 - prediction) * 100, 2)
        st.success("✅ Mask Detected")
        st.progress(int(confidence))
        st.write(f"Confidence: {confidence}%")

st.markdown("---")
st.caption("Built with ❤️ using MobileNetV2 and Streamlit")

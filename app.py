import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="AI Face Mask Detection",
    layout="wide"
)

st.sidebar.title("📦 Project Info")
st.sidebar.markdown("""
**Model:** MobileNetV2  
**Framework:** TensorFlow  
**Face Detection:** OpenCV Haar Cascade  
**Deployment:** Streamlit Cloud  
**Developer:** Shivam 🚀  
""")

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_model_cached():
    return load_model("face_mask_model.keras")

@st.cache_resource
def load_face_detector():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

model = load_model_cached()
face_detector = load_face_detector()

IMG_SIZE = 224

# ---------------------------
# App UI
# ---------------------------
st.title("😷 AI Face Mask Detection")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    if len(faces) > 0:

        for (x, y, w, h) in faces:
            face = img_np[y:y+h, x:x+w]

            face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            face_resized = face_resized / 255.0
            face_resized = np.expand_dims(face_resized, axis=0)

            prediction = model.predict(face_resized, verbose=0)[0][0]

            label = "Mask" if prediction < 0.5 else "No Mask"
            confidence = round(
                float(1 - prediction if prediction < 0.5 else prediction) * 100,
                2
            )

            color = (0, 255, 0) if label == "Mask" else (255, 0, 0)

            cv2.rectangle(img_np, (x, y), (x+w, y+h), color, 3)
            cv2.putText(
                img_np,
                f"{label} ({confidence}%)",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        st.success("Face detected successfully!")
        st.image(img_np, use_container_width=True)

    else:
        st.warning("No face detected.")
        st.image(img_np, use_container_width=True)

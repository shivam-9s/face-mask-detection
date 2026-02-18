import streamlit as st
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw

st.set_page_config(page_title="AI Face Mask Detection", layout="wide")

st.sidebar.title("📦 Project Info")
st.sidebar.markdown("""
**Model:** MobileNetV2  
**Framework:** TensorFlow  
**Face Detection:** MediaPipe  
**Deployment:** Streamlit Cloud  
**Developer:** Shivam 🚀  
""")

@st.cache_resource
def load_model_cached():
    return load_model("face_mask_model.keras")

@st.cache_resource
def load_detector():
    mp_face = mp.solutions.face_detection
    return mp_face.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.3
    )

model = load_model_cached()
detector = load_detector()
IMG_SIZE = 224

st.title("😷 AI Face Mask Detection")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    results = detector.process(img_np)

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

            face_pil = Image.fromarray(face).resize((IMG_SIZE, IMG_SIZE))
            face_array = np.array(face_pil) / 255.0
            face_array = np.expand_dims(face_array, axis=0)

            prediction = model.predict(face_array, verbose=0)[0][0]

            label = "Mask" if prediction < 0.5 else "No Mask"
            confidence = round(float(1 - prediction if prediction < 0.5 else prediction) * 100, 2)

            color = "green" if label == "Mask" else "red"

            draw.rectangle(
                [x, y, x + width, y + height],
                outline=color,
                width=4
            )

            draw.text(
                (x, y - 20),
                f"{label} ({confidence}%)",
                fill=color
            )

        st.success("Face detected successfully!")
        st.image(image, use_container_width=True)

    else:
        st.warning("No face detected.")
        st.image(image, use_container_width=True)

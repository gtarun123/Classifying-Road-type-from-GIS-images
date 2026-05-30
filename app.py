import io
import os

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

CLASS_NAMES = ["Highway", "Street Road", "Rural Road", "Dirt Road", "Flyover Road"]
MODEL_PATH = "models/road_type_classifier.h5"


def load_model(path):
    """Load the trained model once and cache it for the Streamlit app."""
    return tf.keras.models.load_model(path)


def preprocess_uploaded_image(uploaded_file, target_size=(224, 224)):
    """Convert the uploaded file buffer into an OpenCV image array."""
    image_bytes = uploaded_file.read()
    image_np = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    if image is None:
        st.error("Unable to read the uploaded image. Please use a valid image file.")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image.astype("float32") / 255.0
    return image


def predict(image, model):
    """Predict the road type of an uploaded image."""
    # Multiply by 255.0 to compensate for the Rescaling(1.0/255) layer in the model
    input_tensor = np.expand_dims(image * 255.0, axis=0)
    predictions = model.predict(input_tensor)
    predicted_index = int(np.argmax(predictions, axis=1)[0])
    confidence = float(np.max(predictions))
    return CLASS_NAMES[predicted_index], confidence


st.set_page_config(page_title="Road Type Classifier", page_icon="🚦", layout="centered")
st.title("Classifying Road Type from GIS Images")
st.write(
    "Upload a satellite/GIS image and predict whether the road is a Highway, City Road, Village Road, or Dirt Road."
)

if not os.path.exists(MODEL_PATH):
    st.warning(
        "The trained model file is not found. Please run `python train.py` first to generate `models/road_type_classifier.h5`."
    )
else:
    model = load_model(MODEL_PATH)
    uploaded_file = st.file_uploader("Upload a GIS image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = preprocess_uploaded_image(uploaded_file)
        if image is not None:
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.write("---")
            predicted_label, confidence = predict(image, model)
            st.success(f"Predicted Road Type: {predicted_label}")
            st.info(f"Confidence: {confidence * 100:.2f}%")

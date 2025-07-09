import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("satellite_classifier_final.h5")

# Define class names
class_names = ['Cloudy', 'Desert', 'Green_Area', 'Water']

# App title
st.title("🛰️ Satellite Image Classifier")

# File uploader
uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = image.resize((224, 224))  # Match model input size
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    predictions = model.predict(img_array)
    predicted_label = class_names[np.argmax(predictions)]

    # Show result
    st.write(f"🧠 Predicted Class: **{predicted_label}**")
    st.bar_chart(predictions[0])

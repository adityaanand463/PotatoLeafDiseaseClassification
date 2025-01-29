import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from PIL import Image
from home import home
import time 
from collections import Counter
import os 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "potato.h5")



model = tf.keras.models.load_model(MODEL_PATH)

class_names =["Early Blight","Late Blight","Healthy"]
def upload():
    uploaded_file = st.file_uploader("Upload a potato leaf image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        img_array = preprocess_image(img)

        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions[0])


        st.subheader("Prediction Results")
        
        st.write("###  CNN Model")
        st.success(f"Prediction: {class_names[class_idx]}")
        st.info(f"Confidence: {predictions[0][class_idx]*100:.2f}")

def preprocess_image(img):
    img = img.resize((256, 256))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    return img_array


if __name__=="__main__":
    if "page" not in st.session_state:
        st.session_state.page = "home"

    if st.session_state.page == "home":
        home()
    
    st.sidebar.header("Options")
    option = st.sidebar.selectbox("Choose Your Work", ["Upload Image"],index=None)
    
    if(option=="Upload Image"):
        upload()

    
    st.markdown("---")
    st.info("📌 Navigate to different sections using the sidebar.")
    st.write("Made with ❤️ by Aditya Anand")
        
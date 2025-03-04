import streamlit as st
import numpy as np
import cv2
import gdown
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os

def download_model():
    url = "https://drive.google.com/uc?id=1dtQaD0rG9gpJ5IyqvJ3K-SWINJ1E91-e"
    output = "model.h5"
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    return load_model(output)

@st.cache_resource
def load_model_cached():
    return download_model()

model = load_model_cached()

def prepro_(image):
    x, y, z = 224, 224, 3  # التأكد من أبعاد الصورة المطلوبة
    new_image = cv2.resize(image, (x, y))
    
    if new_image.ndim == 2:  # إذا كانت الصورة رمادية، نحولها إلى 3 قنوات
        new_image = cv2.cvtColor(new_image, cv2.COLOR_GRAY2RGB)
    elif new_image.shape[-1] == 4:  # إذا كانت الصورة تحتوي على قناة ألفا (RGBA)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2RGB)
    
    new_image = new_image.astype('float32') / 255.0
    new_image = img_to_array(new_image)
    new_image = new_image.reshape(1, x, y, z)
    return new_image

st.title("Image Classification App")
st.markdown("<br>", unsafe_allow_html=True)
uploaded_image = st.file_uploader("Upload an Image", type=['jpg', 'png', 'jpeg'])

if uploaded_image is not None:
    img = Image.open(uploaded_image)
    img = np.array(img)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    processed_image = prepro_(img)
    if st.button("Predict"):
        prediction = model.predict(processed_image)
        class_index = np.argmax(prediction)
        predicted_class = f"Class {class_index}"  # تعديل لطريقة عرض التوقع
        st.success(f'This image represents: **{predicted_class}**')

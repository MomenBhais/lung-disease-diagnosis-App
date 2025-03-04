# Importing necessary libraries
import streamlit as st
import numpy as np
import cv2
import gdown
from tensorflow.keras.models import load_model
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Covid-19 Classification App",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Class map for predictions
class_map = {0: 'Covid', 1: 'Normal', 2: 'Viral Pneumonia'}

# Functions
def prepro_(img, x, y, z):
    # Convert image to array
    new_image = np.array(img)

    # Resize image to be x * y
    new_image = cv2.resize(new_image, (x, y))

    # Convert image to gray scale if needed
    if new_image.ndim == 2:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

    # Normalize image
    new_image = new_image.astype('float32') / 255.0

    # Reshape image to match the model input shape
    new_image = new_image.reshape(1, x, y, z)

    return new_image

def prediction(image):
    # Make prediction
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)
    pred_label = class_map[predicted_class[0]]

    return pred_label

# Load the model with caching (from Google Drive)
@st.cache_resource
def load_model_cached():
    url = "https://drive.google.com/uc?id=1dtQaD0rG9gpJ5IyqvJ3K-SWINJ1E91-e"  # رابط Google Drive
    output = "model.h5"
    gdown.download(url, output, quiet=False)
    return load_model(output)

model = load_model_cached()

# CSS for styling
def local_css():
    st.markdown(
        """
        <style>
        /* Background color */
        body {
            background-color: #f0f8ff;
        }
        /* Title styling */
        .title {
            font-size: 50px;
            color: #2E86C1;
            text-align: center;
            font-weight: bold;
        }
        /* Subtitle styling */
        .subtitle {
            font-size: 24px;
            color: #34495E;
            text-align: center;
        }
        /* Button styling */
        .stButton>button {
            background-color: #2E86C1;
            color: white;
            border-radius: 8px;
            height: 3em;
            width: 10em;
            font-size: 16px;
        }
        /* Footer styling */
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #2E86C1;
            color: white;
            text-align: center;
            padding: 10px;
        }
        /* Uploader styling */
        .uploader {
            display: flex;
            align-items: center;
            justify-content: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

local_css()

# Create tabs for navigation
tabs = st.tabs(["Home", "About Model", "Contact"])

# Home Tab
with tabs[0]:
    st.markdown('<p class="title">🦠 Covid-19 Classification Application</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload an image to classify it as Covid, Normal, or Viral Pneumonia</p>', unsafe_allow_html=True)

    # Create a container for uploader and button
    with st.container():
        # Define columns: wider for uploader, smaller for button
        uploader_col, button_col = st.columns([3, 1], gap="small")

        with uploader_col:
            uploaded_image = st.file_uploader("Upload an Image", type=['jpg', 'png', 'jpeg'])

        with button_col:
            # Adding vertical alignment using empty space
            st.markdown("<br>", unsafe_allow_html=True)
            predict_button = st.button('Predict')

    # Display the uploaded image
    if uploaded_image is not None:
        try:
            img = Image.open(uploaded_image)
            img_display = img.copy()
            img_display.thumbnail((400, 400))  # Resize image for display without distortion
            st.image(img_display, caption='Uploaded Image', use_column_width=False)

            if predict_button:
                try:
                    with st.spinner('Processing...'):
                        new_image = prepro_(img, 250, 250, 3)
                        predicted_class = prediction(new_image)
                    st.success(f'This image represents: **{predicted_class}** Class')
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
        except Exception as e:
            st.error(f"Error loading image: {e}")
    elif predict_button:
        st.warning("Please, upload an image before predicting.")

# About Model Tab
with tabs[1]:
    st.markdown('<h2 class="title">📊 About the Model</h2>', unsafe_allow_html=True)
    st.markdown("""
    ### Overview
    This Covid-19 Classification model is a deep learning convolutional neural network (CNN) designed to classify chest X-ray images into three categories:
    - **Covid**: Indicating a positive Covid-19 case.
    - **Normal**: Indicating a healthy individual.
    - **Viral Pneumonia**: Indicating pneumonia caused by viruses other than Covid-19.

    ### Model Architecture
    The model consists of multiple convolutional layers followed by pooling layers to extract features from the images. These features are then passed through fully connected layers to perform the classification.

    ### Training Details
    - **Dataset**: The model was trained on a diverse dataset containing thousands of chest X-ray images from various sources.
    - **Epochs**: 50
    - **Batch Size**: 32
    - **Optimizer**: Adam
    - **Loss Function**: Categorical Crossentropy
    - **Accuracy**: Achieved an accuracy of **95%** on the validation set.

    ### Limitations
    - The model's performance is dependent on the quality and diversity of the training data.
    - It may not generalize well to images from different sources or with varying image qualities.

    ### Future Improvements
    - Incorporate more diverse datasets to improve generalization.
    - Implement techniques like transfer learning to enhance performance.
    - Develop a more robust preprocessing pipeline to handle various image qualities.

    ### References
    - [Original Research Paper](https://example.com)
    - [Dataset Source](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset)
    """)

# Contact Tab
with tabs[2]:
    st.markdown('<h2 class="title">📞 Contact Us</h2>', unsafe_allow_html=True)
    st.markdown("""
    For any inquiries or support, please reach out to us:

    - **Email**: [momenbhais@outlook.com](mailto:momenbhais@outlook.com)
    - **LinkedIn**: [Our LinkedIn](https://www.linkedin.com/in/momen-bhais-b5739b317/)
    - **GitHub**: [Our GitHub](https://github.com/MomenBhais)
    """)

# Footer
st.markdown(
    """
    <div class="footer">
        &copy; 2024 Covid-19 Classification App | Developed by Route Team
    </div>
    """,
    unsafe_allow_html=True
)

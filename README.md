# COVID-19 & Lung Disease Detection App

![Python Logo](https://www.python.org/static/community_logos/python-logo.png)

## 📌 Project Overview
The **COVID-19 & Lung Disease Detection App** is an AI-powered application designed to assist in detecting lung diseases, including **COVID-19, Pneumonia, and Normal lung conditions**, from medical images such as chest X-rays. By leveraging deep learning models, the app provides a fast and reliable diagnosis that can support healthcare professionals in decision-making.
##Normal
![Image](https://github.com/user-attachments/assets/4104b8b1-4fd4-461f-b526-9004c458672e)
##Viral Pneumonia
![Image](https://github.com/user-attachments/assets/e4551010-f1ff-4af8-836d-2c9ed1b37273)
##Covid
![Image](https://github.com/user-attachments/assets/21cec8a1-342d-48f3-b06c-d5e9aeb7689a)
## 🚀 Features
- 🏥 **Automated Lung Disease Detection**: Identifies whether the lungs are affected by COVID-19, Pneumonia, or are in a normal state.
- 📊 **Visualization & Reports**: Displays confidence scores and detailed reports for each prediction.
- 🧠 **Deep Learning Model**: Uses Convolutional Neural Networks (CNNs) for high accuracy.
- 🖥️ **User-Friendly Interface**: A web-based UI for easy interaction.
- 📡 **API Support**: Enables seamless integration into healthcare systems.
- 🔄 **Cross-Platform Compatibility**: Works on Windows, macOS, and Linux.

## 🏗️ Tech Stack
- **Programming Language**: Python
- **Frameworks**: TensorFlow / PyTorch, Flask, Streamlit
- **Computer Vision**: OpenCV, NumPy, Matplotlib
- **Deep Learning Model**: Pre-trained CNN architectures (ResNet, VGG, etc.)
- **Frontend**: Streamlit for UI
- **Backend**: Flask or FastAPI
- **Deployment**: Docker / Cloud (AWS, Google Cloud, or Azure)

## ⚙️ Installation & Setup
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- pip (Python package manager)
- Virtual environment (optional but recommended)

### Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/covid19-detection.git
   cd covid19-detection
   ```
2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the application**:
   ```bash
   python app.py
   ```

## 🎯 Usage
- Upload a chest X-ray image.
- The app will process the image and determine if the lungs are **normal, affected by Pneumonia, or COVID-19 positive**.
- View the confidence score and analysis.

## 📂 Project Structure
```
📦 COVID-19 & Lung Disease Detection App
 ┣ 📂 models           # Pre-trained deep learning models
 ┣ 📂 static           # CSS, JS, and image files
 ┣ 📂 templates        # HTML templates (if using Flask)
 ┣ 📂 utils            # Helper scripts for preprocessing
 ┣ 📜 app.py           # Main application file
 ┣ 📜 requirements.txt # Required dependencies
 ┣ 📜 README.md        # Project documentation
```

## 🔬 Code Explanation
### app.py (Main Application File)
- Loads the pre-trained deep learning model.
- Captures input from uploaded images.
- Processes the image using OpenCV and prepares it for the deep learning model.
- Passes the processed image to the model for prediction.
- Displays the results on the web UI.

Example snippet:
```python
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request

app = Flask(__name__)
model = tf.keras.models.load_model('models/lung_disease_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = cv2.imread(file)
    img = cv2.resize(img, (224, 224)) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    result = 'COVID-19' if prediction[0][0] > 0.5 else ('Pneumonia' if prediction[0][1] > 0.5 else 'Normal')
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
```

## 🚀 Deployment
You can deploy the app using Docker or cloud platforms.
### Using Docker:
```bash
docker build -t lung-disease-app .
docker run -p 5000:5000 lung-disease-app
```

## 📌 Future Enhancements
- 🔄 Improve model accuracy with additional datasets.
- 📱 Develop a mobile application version.
- 🌎 Add multilingual support.
- 🎤 Integrate voice-based symptom analysis.

## 🛠️ Contributing
We welcome contributions! Fork this repo and submit a pull request.

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact
For inquiries or support, contact: `momenbhais@outlook.com`

---
⭐ If you find this project helpful, consider giving it a star on GitHub!

**Author**: Momen Mohammed Bhais

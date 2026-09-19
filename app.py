import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Page Title and Instructions
st.title("X-Ray Image Classification App")
st.write("Upload an X-ray image below to test it against the trained VGG16 model.")

# 2. Cache and Load your trained model
@st.cache_resource
def load_model():
    # Make sure 'vgg16_xray_model.keras' is in the same folder as this script
    model = tf.keras.models.load_model('vgg16_xray_model.keras')
    return model

with st.spinner("Loading model... Please wait."):
    model = load_model()

# Define your class names based on your training dataset folders
class_names = ['NORMAL', 'COVID19'] # Update these names if your folder labels differ

# 3. File Uploader widget for users
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image on the web page
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded X-Ray Image', use_column_width=True)
    
    # 4. Preprocess the image for VGG16
    img_resized = image.resize((224, 224))
    x = tf.keras.utils.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.vgg16.preprocess_input(x)
    
    # 5. Predict button and output
    if st.button('Predict Diagnosis'):
        with st.spinner('Analyzing image...'):
            predictions = model.predict(x)
            predicted_class_index = np.argmax(predictions[0])
            predicted_class_name = class_names[predicted_class_index]
            confidence = np.max(predictions[0]) * 100
            
        st.success(f"**Prediction:** {predicted_class_name}")
        st.info(f"**Confidence Score:** {confidence:.2f}%")
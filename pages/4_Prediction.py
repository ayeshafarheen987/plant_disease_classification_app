import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

st.markdown("""
    <style>
        body {
            background-image: url("https://e1.pxfuel.com/desktop-wallpaper/1016/883/desktop-wallpaper-light-teal-color-backgrounds-light-teal-background.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        .stApp {
            background-color: rgba(255, 255, 255, 0.3);
        }
    </style>
""", unsafe_allow_html=True)





# Define class names
class_names = [
    'Apple scab', 'Apple black rot', 'Apple cedar apple rust', 'Apple healthy',
    'Blueberry healthy', 'Cherry powdery mildew', 'Cherry healthy',
    'Corn cercospora leaf spot gray leaf spot', 'Corn common rust',
    'Corn northern leaf blight', 'Corn healthy', 'Grape black rot',
    'Grape esca (black measles)', 'Grape leaf blight (isariopsis leaf spot)', 'Grape healthy',
    'Orange haunglongbing (citrus greening)', 'Peach bacterial spot', 'Peach healthy',
    'Bell pepper bacterial spot', 'Bell pepper healthy', 'Potato early blight',
    'Potato late blight', 'Potato healthy', 'Raspberry healthy', 'Soybean healthy',
    'Squash powdery mildew', 'Strawberry leaf scorch', 'Strawberry healthy',
    'Tomato bacterial spot', 'Tomato early blight', 'Tomato late blight', 'Tomato leaf mold',
    'Tomato septoria leaf spot', 'Tomato spider mites two-spotted spider mite', 'Tomato target spot',
    'Tomato tomato yellow leaf curl virus', 'Tomato tomato mosaic virus', 'Tomato healthy'
]

# Define the function for image classification
def classify_image(image):
    # Preprocess the image
    image = image.resize((224, 224))  # Resize the image to match the model's input shape
    image = np.array(image)  # Convert image to numpy array
    image = image / 255.0  # Normalize the image
    
    # Reshape the image to match the model's input shape
    image = np.reshape(image, (1, 224, 224, 3))
    
    # Load the quantized TensorFlow Lite model
    interpreter = tf.lite.Interpreter(model_path='pages/efficientnet_quantized.tflite')
    interpreter.allocate_tensors()
    
    # Get the input and output details of the model
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set the input tensor
    input_index = input_details[0]['index']
    interpreter.set_tensor(input_index, image.astype(np.float32))
    
    # Run the inference
    interpreter.invoke()
    
    # Get the output tensor
    output_index = output_details[0]['index']
    predictions = interpreter.get_tensor(output_index)
    
    # Get the predicted class and confidence
    class_index = np.argmax(predictions)
    class_name = class_names[class_index]
    confidence = predictions[0][class_index] * 100
    
    return class_name, confidence

# Set page title
st.markdown("<h1 style='color: red;'>Plant Disease Classification</h1>", unsafe_allow_html=True)

# Create the file uploader
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image')
    
    # Classify the image
    class_name, confidence = classify_image(image)
    
    # Display the result
    st.write(f"Class: {class_name}")
    st.write(f"Confidence: {confidence:.2f}%")

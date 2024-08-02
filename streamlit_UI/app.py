import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the model using keras directly from tensorflow namespace
model_path = '../ASL/best_model_2.h5'
model = tf.keras.models.load_model(model_path, compile=False)

# Define class labels based on test output
class_labels = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
    25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'
}

def process_image(frame):
    img = Image.open(frame).convert('L')
    img = img.resize((64, 64))
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=(0, -1))  # Add batch and channel dimensions
    return img_array

def predict_frame(frame, model, class_labels):
    try:
        img_array = process_image(frame)
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class = class_labels[predicted_class_index]
        return predicted_class
    except Exception as e:
        st.error(f"Error processing frame: {e}")
        return "Error"

st.title('ASL Recognition using Webcam')

# Initialize session state to store the recognized string
if 'recognized_string' not in st.session_state:
    st.session_state['recognized_string'] = ""

# Setup the webcam capture
cap = st.camera_input("Capture", key="cam")

if cap:
    # If a frame is captured, predict the class
    predicted_class = predict_frame(cap, model, class_labels)
    
    # If 'del' is predicted, remove the last character
    if predicted_class == 'del':
        st.session_state['recognized_string'] = st.session_state['recognized_string'][:-1]
    elif predicted_class == 'space':
        st.session_state['recognized_string'] += ' '
    elif predicted_class != 'nothing':
        st.session_state['recognized_string'] += predicted_class
    
    st.write('Predicted Sign Language: ', predicted_class)
    st.write('Recognized String: ', st.session_state['recognized_string'])

# To reset the recognized string
if st.button('Reset'):
    st.session_state['recognized_string'] = ""
    st.write('Recognized String has been reset.')

# To stop using the webcam
if st.button('Stop'):
    st.write("Webcam stopped.")

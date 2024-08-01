import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the model
model_path = '../ASL/best_model_2.h5'
model = load_model(model_path)

# Define class labels based on test output
class_labels = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'H': 7,
    'I': 8,
    'J': 9,
    'K': 10,
    'L': 11,
    'M': 12,
    'N': 13,
    'O': 14,
    'P': 15,
    'Q': 16,
    'R': 17,
    'S': 18,
    'T': 19,
    'U': 20,
    'V': 21,
    'W': 22,
    'X': 23,
    'Y': 24,
    'Z': 25,
    'del': 26,
    'nothing': 27,
    'space': 28
}


def predict_frame(frame, model, class_labels):
    try:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        img = cv2.resize(img, (64, 64))  # Resize to model expected size
        img_array = img_to_array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class = class_labels[predicted_class_index]
        return predicted_class
    except Exception as e:
        print(f"Error processing frame: {e}")
        return "Error"


# Streamlit webpage title
st.title('ASL Recognition using Webcam')

# Setting up the webcam
FRAME_WINDOW = st.image([])
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Make predictions on the current frame
    predicted_class = predict_frame(frame, model, class_labels)

    # Display the camera feed
    FRAME_WINDOW.image(frame, channels="BGR")

    # Display the prediction
    st.write('Predicted Sign Language: ', predicted_class)

    # Break the loop with a button in Streamlit
    if st.button('Stop'):
        break

# Release the camera and close all windows
cap.release()

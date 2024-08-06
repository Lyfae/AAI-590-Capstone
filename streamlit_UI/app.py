import streamlit as st
import numpy as np
import os
import tensorflow as tf
import torch
from PIL import Image
from peft import PeftModel
from transformers import GenerationConfig, AutoModelForSeq2SeqLM, AutoTokenizer

# Load the model using keras directly from tensorflow namespace
model_path = '../ASL/best_model_3.h5'
model = tf.keras.models.load_model(model_path, compile=False)

peft_model_base = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

peft_model = PeftModel.from_pretrained(peft_model_base,
                                       '../flan_t5/peft-conversation-checkpoint-local',
                                       torch_dtype=torch.bfloat16,
                                       is_trainable=False)

# Image path
image_path = './images'

# Define class labels based on test output
class_labels = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
    25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'
}

def insert_prompt(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids 

    peft_model_outputs = peft_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
    peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)

    return peft_model_text_output

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

def text_to_images(response, path):
    # Take the response string, make it into a list and capitalize it
    response = list(response.upper())

    # Empty list to hold the images
    images = []

    # Iterate through the responses and look for the files associated with the name
    for char in response:
        if char == ' ':
            char = 'space'
        image_name = f"{char}.jpg"
        image_path = os.path.join(path,image_name)

        # If file exists, append to our list
        if os.path.isfile(image_path):
            images.append(image_path)
        else:
            print("File was not found in path")

    return images

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


# Create columns for buttons
col1, col2 = st.columns(2)
# Buttons
with col1:
    # To reset the recognized string    
    if st.button('Reset'):
        st.session_state['recognized_string'] = ""
        st.write('Recognized String has been reset.')
with col2:
    # To stop using the webcam
    if st.button('BackSpace'):
        st.session_state['recognized_string'][:-1]

# To stop using the webcam
if st.button('Stop'):
    st.write("Webcam stopped.")

if st.button('Ask Bot'):
    st.write("User's Prompt: ", "What is the capital of california") # change this to st.session_state['recognized_string'] later
    answer = insert_prompt("What is the capital of california") # change this to st.session_state['recognized_string'] later
    st.write("Bot's Repsonse: ", answer)
    visual_answer = text_to_images(answer, image_path)
    st.image(visual_answer, width=100)
    

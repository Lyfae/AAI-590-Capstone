# Import essential libraries and modules
import streamlit as st  # Streamlit library for creating web apps
import numpy as np  # NumPy library for numerical operations
import os  # OS module to interact with the operating system
import tensorflow as tf  # TensorFlow library for deep learning models
import torch  # PyTorch library for tensor and neural network operations
from PIL import Image  # PIL library for image processing
from peft import PeftModel  # PEFT library for fine-tuned language models
from transformers import GenerationConfig, AutoModelForSeq2SeqLM, AutoTokenizer  # Hugging Face's transformers for NLP models

# Load the TensorFlow Keras model from the specified path without compiling
model_path = '../ASL/asl_cnn_model.h5'
model = tf.keras.models.load_model(model_path, compile=False)

# Load a sequence-to-sequence language model and tokenizer from Hugging Face
peft_model_base = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

# Configure and load a fine-tuned version of the PEFT model with non-trainable parameters
peft_model = PeftModel.from_pretrained(peft_model_base,
                                       '../flan_t5/peft-conversation-checkpoint-local',
                                       torch_dtype=torch.bfloat16,
                                       is_trainable=False)

# Directory to store and fetch images
image_path = './images'

# Mapping from numerical labels to corresponding ASL alphabet signs
class_labels = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
    25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'
}

def insert_prompt(prompt):
    # Prepare the prompt for the PEFT model and generate responses
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    peft_model_outputs = peft_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
    peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)
    return peft_model_text_output

def process_image(frame):
    # Process image input for prediction: grayscale, resize, normalize, and reshape
    img = Image.open(frame).convert('L')
    img = img.resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))
    return img_array

def predict_frame(frame, model, class_labels):
    # Use the processed image to predict the ASL sign using the loaded model
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
    # Convert recognized text into corresponding images from a predefined directory
    response = list(response.upper())
    images = []
    for char in response:
        char = 'space' if char == ' ' else char
        image_name = f"{char}.jpg"
        image_path = os.path.join(path, image_name)
        if os.path.isfile(image_path):
            images.append(image_path)
        else:
            print("File was not found in path")
    return images

st.title('ASL Recognition using Webcam')

# Initialize a session state to keep track of the recognized string across sessions
if 'recognized_string' not in st.session_state:
    st.session_state['recognized_string'] = ""

# Provide an interface for webcam input
cap = st.camera_input("Capture", key="cam")

if cap:
    # Predict ASL sign from captured image
    predicted_class = predict_frame(cap, model, class_labels)
    # Manage special cases like 'delete' and 'space'
    if predicted_class == 'del':
        st.session_state['recognized_string'] = st.session_state['recognized_string'][:-1]
    elif predicted_class == 'space':
        st.session_state['recognized_string'] += ' '
    elif predicted_class != 'nothing':
        st.session_state['recognized_string'] += predicted_class
    
    st.write('Predicted Sign Language: ', predicted_class)
    st.write('Recognized String: ', st.session_state['recognized_string'])

# Layout for control buttons to manage the recognized string
col1, col2, col3 = st.columns(3)
with col1:
    if st.button('Reset'):
        st.session_state['recognized_string'] = ""
        st.write('Recognized String has been reset.')
with col2:
    if st.button('BackSpace'):
        st.session_state['recognized_string'] = st.session_state['recognized_string'][:-1]
with col3:
    if st.button('Question Mark'):
        st.session_state['recognized_string'] += '?'
st.write("Updated String: ", st.session_state['recognized_string'])

# Additional controls for user interaction
if st.button('Stop'):
    st.write("Webcam stopped.")

if st.button('Ask Bot'):
    st.write("User's Prompt: ", st.session_state['recognized_string'])
    answer = insert_prompt(st.session_state['recognized_string'])
    st.write("Bot's Response: ", answer)
    visual_answer = text_to_images(answer, image_path)
    st.image(visual_answer, width=100)

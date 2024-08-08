# AAI-590-Captstone

## Overview
This project is designed to provide an interactive web application for American Sign Language (ASL) recognition using a webcam. The application is built using Streamlit and integrates machine learning models from TensorFlow and language models from Hugging Face's Transformers library. Users can capture ASL signs via a webcam, and the application will predict and display the corresponding alphabet or action.

## Getting Started
Follow these instructions to set up the project environment and run the application on your local machine.


### Prerequisites
Python 3.8 or later
pip
Virtual environment (recommended)


### Installation
1 . Clone the Repository

Start by cloning the repository to your local machine:

git clone < https://github.com/Lyfae/AAI-590-Capstone.git >
cd AAI-590-Captstone


2. Create a Virtual Environment (Optional, but recommended)


Create a virtual environment to manage the dependencies for the project:

python -m venv venv

Activate the virtual environment:

On Windows:
.\venv\Scripts\activate

On macOS/Linux:
source venv/bin/activate

3. Install Dependencies 

Install the required packages from the requirements.txt file:

pip install -r requirements.txt

### Running the Application
To run the Streamlit application, navigate to the streamlit_UI directory and execute the following command:

streamlit run app.py

Make sure you are in the correct directory that contains the app.py file.


## Features

### Webcam Integration: 
Capture real-time video through the webcam to recognize ASL signs.
## ASL Recognition: 
Utilizes a trained deep learning model to classify ASL signs from the webcam input.
### Interactive Interface: 
Allows users to interact with the application, view predictions, and modify recognized text.
### Text-to-Sign Translation: 
Converts textual input back into ASL signs displayed as images.
### Dialogue Interface: 
Includes a bot that can respond to text generated from ASL signs, enhancing the user interaction.

## Files and Directories
app.py: The main application file where the Streamlit UI is defined.
requirements.txt: A file listing all the dependencies required to run the project.

## How It Works
### ASL Recognition: 
The application processes video input from the webcam, recognizes ASL signs, and translates them into text.
### Text Handling: 
The recognized text can be edited via the application's interface, allowing users to correct or modify the output.
### Interaction: 
Users can engage with an in-built bot that processes the recognized or input text and provides intelligent responses.

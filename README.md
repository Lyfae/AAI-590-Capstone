# AAI-590-Captstone

## Overview
This project aims at developing an application able to elaborate live sign language questions and return both textual and sign language answers on general topics, thereby offering communication support to deaf and dumb individuals.

## Getting Started
Follow these instructions to set up the project environment and run the application on your local machine.


### Prerequisites
Python 3.8 or later  
pip  
Virtual environment (recommended)

**NOTE: The environment used for this project ran on used Python 3.9.15**

### Create Conda Environment
If you have conda , please use the following to setup your environement:

```
conda create -n "myenv" python=3.9.15 ipython
```

This will create a environment named `myenv` and will utilize Python 3.9.15 with the ipython package installed. 

### Installation

1. Clone the Repository
    - Start by cloning the repository to your local machine:

    - git clone < https://github.com/Lyfae/AAI-590-Capstone.git >
    - ``cd AAI-590-Captstone``

2. Create a Virtual Environment (Optional, but recommended)
    - Create a virtual environment to manage the dependencies for the project:
        - ``python -m venv venv``

    - Activate the virtual environment:

    - On Windows:
        - ``.\venv\Scripts\activate``

    - On macOS/Linux:
        - ``source venv/bin/activate``

3. Install Dependencies 

    - Install the required packages from the requirements.txt file:

        - ``pip install -r requirements.txt``

4. Optional - Setup Hugging Face Hub to access flan-t5-base model

    - Install the huggingface library `if you already don't have it on your system already: should be included in requirements.txt`

        - ``pip install --upgrade huggingface_hub``
    - Inisde the terminal, authenticate yourself with the following using a token from Hugging face:
        - ``huggingface-cli login``
        - For more information on creating a token, visit the link [here](https://huggingface.co/docs/hub/en/security-tokens)
        - For more information about the Hugging Face Cli, visit the link [here](https://huggingface.co/docs/huggingface_hub/en/quick-start)

    - Once you have this all setup, you should be good to go.

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
The application processes video input from the webcam, recognizes ASL signs, and translates them into text. Please ensure the image is cleared before utilizing any other buttons. 
### Text Handling: 
The recognized text can be edited via the application's interface, allowing users to correct or modify the output.
### Interaction: 
Users can engage with an in-built bot that processes the recognized or input text and provides intelligent responses.

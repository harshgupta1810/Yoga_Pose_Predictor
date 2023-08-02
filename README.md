# Project Name: Yoga Pose Image Classification

## Table of Contents
1. [Introduction](#introduction)
2. [Project Description](#project-description)
3. [Technologies Used](#Technologies-Used)
4. [Badges](#badges)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Configuration](#configuration)
8. [Contributing](#contributing)
9. [License](#license)
10. [Acknowledgments](#acknowledgments)
11. [Documentation](#documentation)

## Introduction
Welcome to the Yoga Pose Image Classification project! This project aims to classify different yoga poses from images using machine learning techniques.

## Project Description
This project has two parts: the backend and the frontend. The backend handles the model training and image classification, while the frontend provides a user-friendly interface to interact with the trained model.

### Dataset
The model is trained on the dataset named 'yoga-pose-image-classification-dataset' provided by Shruti Saxena. This dataset contains images of various yoga poses along with their corresponding labels.

### Backend
The backend code provided here performs the following tasks:

1. Data Preprocessing: The code uses the ImageDataGenerator from TensorFlow to perform data augmentation and preprocessing. This ensures that the model is trained on diverse and normalized image data.

2. Model Architecture: The backend code utilizes the InceptionResNetV2 model as the base model and adds custom classification layers on top of it. The model is fine-tuned to classify yoga poses.

3. Model Training: The code trains the model using the training data and evaluates its performance on the validation data. It also employs various callbacks, such as ReduceLROnPlateau and EarlyStopping, to optimize the training process.

4. Model Saving: Once the training is complete, the model is saved to a file named 'Yoga_model_1.h5' for later use.

5. Image Prediction: The backend code includes a function that can be used to predict the yoga pose from a given image using the trained model.

### Frontend
The frontend code consists of three files: 'app.py', 'index.html', and 'script.js'.

app.py
This is the Flask application that serves as the backend for the frontend. It loads the trained model, defines the label names for yoga poses, and handles the prediction endpoint. It uses the TensorFlow model to predict the yoga pose from the uploaded image and returns the result as JSON.

index.html
This HTML file is the user interface for the application. It contains a form with an input field to upload an image and a 'Predict' button. When the user selects an image and clicks the button, it triggers the prediction process.

script.js
This JavaScript file handles the form submission, image upload, and communication with the Flask backend. It uses the Fetch API to send the image data to the backend for prediction and displays the result on the webpage.


## Technologies Used

### Algorithms:
1. Convolutional Neural Network (CNN): A deep learning algorithm commonly used for image classification tasks. In this project, a CNN is used to build the image classification model for recognizing yoga poses from input images.

2. Transfer Learning: Transfer learning is a technique where a pre-trained neural network model (InceptionResNetV2) is used as a base model, and custom classification layers are added on top of it. The pre-trained model is fine-tuned on the yoga pose image dataset to improve its performance on the specific classification task.

3. Data Augmentation: Data augmentation is a data preprocessing technique used to artificially increase the size of the training dataset. It involves applying random transformations such as rotation, flipping, and zooming to the input images. Data augmentation helps the model generalize better to new and unseen data.
### Technologies:
1. Python: The core programming language used for the backend development, model training, and data processing.

2. TensorFlow: An open-source deep learning library developed by Google. TensorFlow is used to build, train, and evaluate the CNN model for yoga pose classification.

3. Keras: A high-level neural networks API, written in Python and capable of running on top of TensorFlow. Keras is used for building and configuring the CNN layers.

4. Flask: A micro web framework in Python used for the backend development. Flask serves the frontend application, handles image prediction requests, and communicates with the trained model.

5. HTML: Hypertext Markup Language is used to create the structure of the frontend user interface.

6. CSS: Cascading Style Sheets is used to add styling and improve the visual appearance of the frontend.

7. JavaScript: The frontend application uses JavaScript to handle form submission, image upload, and asynchronous communication with the Flask backend.

8. Matplotlib: A plotting library in Python used for creating various types of visualizations, such as line plots and pie charts. It is used to visualize the sentiment analysis results in the backend code.

9. Numpy: A powerful library for numerical computing in Python. It is used for various numerical operations in the backend code.

10. PIL (Python Imaging Library): A library in Python used for opening, manipulating, and saving image files. It is used for image preprocessing in the backend code.

These algorithms and technologies work together to create a Yoga Pose Image Classification project with both backend and frontend components. The backend handles the CNN model training, sentiment analysis on news articles, and prediction of yoga poses from input images, while the frontend provides a user-friendly interface for users to upload images and receive predictions.

## Badges
[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Installation
To use the Yoga Pose Image Classification project, follow these steps:

1. Install Python: Download and install Python from the official website: https://www.python.org/downloads/

2. Install the required libraries: Open a command prompt or terminal and run the following commands to install the necessary libraries:
   ```
   pip install numpy tensorflow keras livelossplot pillow matplotlib
   ```

## Usage
1. Download the 'yoga-pose-image-classification-dataset' from the provided link and place it in a directory named 'dataset' at the same level as the backend code file.

2. Run the backend code to preprocess the data, train the model, and save it as 'Yoga_model_1.h5'.

3. For the frontend part, create a web application using web development technologies like HTML, CSS, and JavaScript. Use a web framework like Flask or Django to interact with the trained model and display the predictions.

## Configuration
The backend code uses the TensorFlow library with mixed-precision policy ('mixed_float16') enabled for better performance on supported hardware.

## Contributing
If you find any issues or have suggestions for improvement, please feel free to submit a pull request or open an issue on the GitHub repository.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
Special thanks to Harsh Gupta for creating this project.

## Documentation
For more details on the code implementation and function usage, refer to the code comments and documentation in the source files.

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow import keras
import numpy as np
import traceback
import os
from PIL import Image

app = Flask(__name__, static_url_path='', static_folder='static')

# Load the trained model
model = tf.keras.models.load_model('./final_models/Yoga_model_1.h5')

# Define the label names
traindir = r'D:\python projects\pose_Estimation\yoga_pose\2d_yoga_full\dataset\Train'
yoga_labels = {i: folder_name for i, folder_name in enumerate(sorted(os.listdir(traindir)))}

# Route for the home page
@app.route('/')
def home():
    return app.send_static_file('index.html')

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        img = Image.open(file).convert('RGB')
        img = img.resize((224, 224))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the image
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        class_name = yoga_labels[class_index]

        return jsonify({'result': class_name})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': 'An error occurred during prediction.'}), 500



if __name__ == '__main__':
    app.run(debug=True)

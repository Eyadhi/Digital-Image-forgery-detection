import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load the saved model
saved_model_path = 'model2.h5'
loaded_model = load_model(saved_model_path)

# Function to preprocess an image for prediction
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.xception.preprocess_input(img_array)
    return img_array

# Function to make predictions
def predict_image(model, image_path):
    preprocessed_img = preprocess_image(image_path)
    prediction = model.predict(preprocessed_img)
    predicted_class = np.argmax(prediction, axis=1)

    if predicted_class == 0:
        return "Fake"
    elif predicted_class == 1:
        return "Real"
    else:
        return "Unknown"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction_result='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', prediction_result='No selected file')

    if file:
        file_path = 'uploads/' + file.filename
        file.save(file_path)
        prediction_result = predict_image(loaded_model, file_path)
        return render_template('index.html', prediction_result=f'The chosen image is predicted as {prediction_result}')


if __name__ == '__main__':
    app.run(debug=True,port=5800)

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

mlp_model = tf.keras.models.load_model('mlp_model.h5')
resnet_model = tf.keras.models.load_model('resnet_model.h5')
cnn_model = tf.keras.models.load_model('cnn_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        model_type = request.form['model']

        img = Image.open(io.BytesIO(file.read()))
        img = img.resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        if model_type == 'mlp':
            model = mlp_model
        elif model_type == 'resnet':
            model = resnet_model
        else:
            model = cnn_model

        prediction = model.predict(img)
        predicted_class = np.argmax(prediction, axis=1)

        if predicted_class == 0:
            result = "Not wearing glasses"
        else:
            result = "Wearing glasses"

        return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)

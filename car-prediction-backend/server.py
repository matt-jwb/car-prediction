import joblib
from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS
import tensorflow as tf

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model('./tf_model')

preprocessor = joblib.load('./preprocessor.pkl')


@app.route('/')
def home():
    return 'POST requests can be made to /predict to make use of the Prediction Model'


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    data_frame = pd.DataFrame(data, index=[0])

    data_frame['age'] = pd.to_numeric(data_frame['age'], errors='coerce')
    data_frame['miles'] = pd.to_numeric(data_frame['miles'], errors='coerce')
    data_frame['num_owner'] = pd.to_numeric(data_frame['num_owner'], errors='coerce')

    data_frame['age_miles'] = data_frame['age'] * data_frame['miles']

    preprocessed_input = preprocessor.transform(data_frame)
    predictions = model.predict(preprocessed_input)

    predicted_price = float(predictions[0][0])

    return jsonify({'predicted_price': predicted_price})


if __name__ == '__main__':
    app.run(port=4444)

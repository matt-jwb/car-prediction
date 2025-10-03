import joblib
from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS
import torch
import numpy as np

from model import CarPriceModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = Flask(__name__)
CORS(app)

preprocessor = joblib.load('./preprocessor.pkl')

input_data = pd.DataFrame({
    'make': ['Volkswagen'],
    'model': ['Jetta'],
    'age': [13],
    'body_type': ['saloon'],
    'miles': [146000],
    'age_miles': [1898000]
})
input_dim = preprocessor.transform(input_data).shape[1]


model = CarPriceModel(input_dim)
model.load_state_dict(torch.load('./car_price_model.pt', map_location=device))
model.to(device)
model.eval()

@app.route('/')
def home():
    return 'POST requests can be made to /predict to make use of the Prediction Model'


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    data_frame = pd.DataFrame(data, index=[0])

    data_frame['age'] = pd.to_numeric(data_frame['age'], errors='coerce')
    data_frame['miles'] = pd.to_numeric(data_frame['miles'], errors='coerce')

    data_frame['age_miles'] = data_frame['age'] * data_frame['miles']

    preprocessed_input = preprocessor.transform(data_frame)
    input_tensor = torch.tensor(preprocessed_input, dtype=torch.float32).to(device)

    with torch.no_grad():
        prediction_log = model(input_tensor).cpu().numpy()
    predicted_price = float(np.expm1(prediction_log).item())

    return jsonify({'predicted_price': predicted_price})


if __name__ == '__main__':
    app.run(port=4444)

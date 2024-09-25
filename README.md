# Car prediction
This repository includes a frontend and backend. The backend is comprised of a python script to train the AI model and a Flask Server to use it. The front end is a React web page which makes a fetch request to the Flask Server and displays the prediction.

## NOTICE
This project is a work in progress and will be recieving updates soon. I aim to improve the accuracy of the model and the user experience of the front end.

## Usage
To use the exiting model just run server.py. Requests can be made to the server through the front end application. train_model.py is used to create new versions of the model. This will require the CSV file which is currently not part of

## Existing model
The current model uses TensorFlow version 2.17.0 and Keras 3.5.0

## Dependencies
```
flask
flask-cors
joblib
keras
pandas
scikit-learn
tensorflow
```

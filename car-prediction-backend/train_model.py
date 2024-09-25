import joblib
import tensorflow as tf
import keras_tuner as kt
import pandas as pd
from tensorflow import keras
from keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

data = pd.read_csv("all_car_adverts_cleaned3.csv")
data['age_miles'] = data['age'] * data['miles']

X = data[['make', 'model', 'age', 'body_type', 'miles', 'num_owner', 'age_miles']]
y = data['car_price']
X_training, X_testing, y_training, y_testing = train_test_split(X, y, test_size=0.2, random_state=66)

numerical_features = ['age', 'miles', 'num_owner', 'age_miles']
categorical_features = ['make', 'model', 'body_type']

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)
    ]
)

# Fit the preprocessor and transform the data
# preprocessor.pkl is needed by the Flask Server
X_train_transformed = preprocessor.fit_transform(X_training)
X_train_transformed = X_train_transformed.toarray() if hasattr(X_train_transformed, 'toarray') else X_train_transformed
X_test_transformed = preprocessor.transform(X_testing)
X_test_transformed = X_test_transformed.toarray() if hasattr(X_test_transformed, 'toarray') else X_test_transformed
joblib.dump(preprocessor, './preprocessor.pkl')

# Define, compile and train the model
model = keras.Sequential([
    keras.layers.Dense(256, activation='relu', input_dim=X_train_transformed.shape[1], kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)
])
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.001))

training = model.fit(X_train_transformed, y_training, validation_data=(X_test_transformed, y_testing), epochs=200, batch_size=128, verbose=1)

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)  # Early stopping
mse = model.evaluate(X_test_transformed, y_testing, verbose=0, callbacks=[early_stopping])
print("Mean Squared Error:", mse)

# Test predict
input_data = pd.DataFrame({
    'make': ['Volkswagen', 'Ford'],
    'model': ['Jetta', 'Focus'],
    'age': [13, 4],
    'body_type': ['saloon', 'hatchback'],
    'miles': [146000, 30546],
    'age_miles': [1898000, 122184]
})
preprocessed_input = preprocessor.transform(input_data)
preprocessed_input = preprocessed_input.toarray() if hasattr(preprocessed_input, 'toarray') else preprocessed_input

predictions = model.predict(preprocessed_input)
print("Predicted Prices:")
for prediction in predictions:
    print(prediction)

# Save the model to be used by the Flask Server
tf.keras.models.save_model(model, './tf_model')

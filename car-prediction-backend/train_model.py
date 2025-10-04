import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import copy
from torch.utils.data import DataLoader, TensorDataset
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from model import CarPriceModel

# =====================
# Use GPU if available
# =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")


# =============
# Prepare data
# =============
data = pd.read_csv("all_car_adverts_cleaned3.csv")
data['age_miles'] = data['age'] * data['miles']

X = data[['make', 'model', 'age', 'body_type', 'miles', 'age_miles']]
y = data['car_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=66)


# ==============
# Preprocessing
# ==============
numerical_features = ['age', 'miles', 'age_miles']
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
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)
joblib.dump(preprocessor, './preprocessor.pkl')


# ========
# Tensors
# ========
# Convert to torch tensor (required for pyTorch)

y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)
y_train_tensor = torch.tensor(y_train_log.values, dtype=torch.float32).view(-1,1)
y_test_tensor = torch.tensor(y_test_log.values, dtype=torch.float32).view(-1,1)

X_train_tensor = torch.tensor(X_train_transformed, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_transformed, dtype=torch.float32)


# Dataset and DataLoaders used for ease and performance
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)


# ======
# Model
# ======
model = CarPriceModel(input_dim=X_train_tensor.shape[1]).to(device)


# =========
# Training
# =========
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_loss = np.inf
best_model = copy.deepcopy(model.state_dict())
no_improvement = 0

for epoch in range(200):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    avg_train_loss = running_loss / len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
    val_loss /= len(test_loader.dataset)

    print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")

    # Early stopping
    if val_loss < best_loss:
        best_loss = val_loss
        best_model = copy.deepcopy(model.state_dict())
        no_improvement = 0
    else:
        no_improvement += 1
        if no_improvement >= 10:
            print("Early stopping")
            break
model.load_state_dict(best_model)


# ===========
# Evaluation
# ===========
model.eval()
with torch.no_grad():
    X_test_tensor_device = X_test_tensor.to(device)
    y_test_tensor_device = y_test_tensor.to(device)
    predictions = model(X_test_tensor_device)
    mse = criterion(predictions, y_test_tensor_device).item()
print("Mean Squared Error:", mse)


# =============
# Test predict
# =============
input_data = pd.DataFrame({
    'make': ['Volkswagen', 'Ford'],
    'model': ['Jetta', 'Focus'],
    'age': [13, 4],
    'body_type': ['saloon', 'hatchback'],
    'miles': [146000, 30546],
    'age_miles': [1898000, 122184]
})
# Expected results are: ~2750 and ~11490

preprocessed_input = preprocessor.transform(input_data)
preprocessed_input_tensor = torch.tensor(preprocessed_input, dtype=torch.float32).to(device)

with torch.no_grad():
    prediction_log = model(preprocessed_input_tensor).cpu().numpy()
prediction = np.expm1(prediction_log)

print("\nPredicted Prices:")
for p in prediction:
    print(f"£{p[0]:,.2f}")

# Save the model to be used by the Flask Server
torch.save(model.state_dict(), './car_price_model.pt')

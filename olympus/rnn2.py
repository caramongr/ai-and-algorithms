import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

# Load data
data = pd.read_excel('SummaryDataFile.xlsx')
data['Departure_Arrival'] = data['Departure_Arrival'].map({'GBSOU - USNYC': 0, 'USNYC - GBSOU': 1})

# Data preprocessing
# Assuming you have time series or sequential data suitable for RNN
features = data.drop(['Time_Started', 'Time_Ended','Total_Fuel_Consumption', 'Avg Fuel Consumption', 'QM2_Boiler_Aft_Usage_Mass_Flow','QM2_Boiler_Fwd_Usage_Mass_Flow', 'QM2_DG01_Usage_Mass_Flow',
    'QM2_DG02_Usage_Mass_Flow',
    'QM2_DG03_Usage_Mass_Flow',
    'QM2_DG04_Usage_Mass_Flow',
    'QM2_GT01_Usage_Mass_Flow',
    'QM2_GT02_Usage_Mass_Flow',
    # 'QM2_Val_POD01_Power',
    # 'QM2_Val_POD02_Power',
    # 'QM2_Val_POD03_Power',
    # 'QM2_Val_POD04_Power'
    ], axis=1)  # Drop non-feature columns
target = data['Total_Fuel_Consumption']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape features for RNN input
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Build RNN model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(1, X_train_scaled.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train_scaled, y_train, epochs=100, verbose=0)

history = model.fit(X_train_scaled, y_train, epochs=100, verbose=0, validation_data=(X_test_scaled, y_test))

# Evaluate the model
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse}')

# Predict function
def predict_rnn(input_features):
    input_df = pd.DataFrame([input_features])
    input_scaled = scaler.transform(input_df)
    input_scaled = input_scaled.reshape((1, 1, input_scaled.shape[1]))
    return model.predict(input_scaled)[0][0]

# Example usage
input_features = {
   
    # 'QM2_NAV_Latitude': 44.0,
    # 'QM2_NAV_Longitude': -60.0,
    'QM2_Val_POD01_Power': 6.5,
    'QM2_Val_POD02_Power': 6.6,
    'QM2_Val_POD03_Power': 6.6,
    'QM2_Val_POD04_Power': 6.6,
    'QM2_Ship_Outside_Pressure': 1000,
    'QM2_NAV_STW_Longitudinal': 24.0,
    'QM2_Ship_Outside_Temperature': 15,
    'Distance_nautical_miles': 3160,
    'Departure_Arrival':1,
    'Year':2023
}


predicted_consumption = predict_rnn(input_features)
print(f"Predicted Total Fuel Consumption: {predicted_consumption}")


# Actual vs Predicted graph
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Total Fuel Consumption')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.show()

# Loss Curve graph
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss (MSE) Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
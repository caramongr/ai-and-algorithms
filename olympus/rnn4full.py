import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model

# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Load data
data = pd.read_excel('1hr-step-MasterDataFile.xlsx')
data['Departure_Arrival'] = data['Departure_Arrival'].map({'GBSOU - USNYC': 0, 'USNYC - GBSOU': 1})

# Filter data
data = data[data['QM2_NAV_STW_Longitudinal'] >= 8]
data = data.dropna()  # Drop rows with NaN values

# Selecting specific features for training
feature_columns = ['QM2_Ship_Outside_Pressure', 
                   'QM2_NAV_STW_Longitudinal', 'QM2_Ship_Outside_Temperature', 
                   'Distance_nautical_miles', 'Departure_Arrival', 'Year',
                'QM2_Boiler_Aft_Usage_Mass_Flow',
                'QM2_Boiler_Fwd_Usage_Mass_Flow',
                'QM2_DG01_Usage_Mass_Flow',
                'QM2_DG02_Usage_Mass_Flow',
                'QM2_DG03_Usage_Mass_Flow',
                'QM2_DG04_Usage_Mass_Flow',
                'QM2_GT01_Usage_Mass_Flow',
                'QM2_GT02_Usage_Mass_Flow',
                'QM2_Val_POD01_Power',
                'QM2_Val_POD02_Power',
                'QM2_Val_POD03_Power',
                'QM2_Val_POD04_Power'
                   ]

features = data[feature_columns]
target = data['Total_Fuel_Consumption']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

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
history = model.fit(X_train_scaled, y_train, epochs=400, verbose=0, validation_data=(X_test_scaled, y_test))

# Evaluate the model
y_pred = model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f'RMSE: {rmse}')
print(f'R2: {r2}')

# Function for prediction using specific input features
def predict_rnn(input_features):
    input_df = pd.DataFrame([input_features], columns=feature_columns)
    input_scaled = scaler.transform(input_df)
    input_scaled = input_scaled.reshape((1, 1, input_scaled.shape[1]))
    return model.predict(input_scaled)[0][0]

# Example Usages
input_features = {
    'QM2_Boiler_Aft_Usage_Mass_Flow': 0.174,
    'QM2_Boiler_Fwd_Usage_Mass_Flow': 0.13,
    'QM2_DG01_Usage_Mass_Flow': 2.4,
    'QM2_DG02_Usage_Mass_Flow': 2.51,
    'QM2_DG03_Usage_Mass_Flow': 2.51,
    'QM2_DG04_Usage_Mass_Flow': 2.51,
    'QM2_GT01_Usage_Mass_Flow': 0,
    'QM2_GT02_Usage_Mass_Flow': 0,
    # 'QM2_NAV_Latitude': 42.0,
    # 'QM2_NAV_Longitude': -60.0,
    'QM2_Ship_Outside_Pressure': 1000,
    'QM2_Val_POD01_Power': 8,
    'QM2_Val_POD02_Power': 8.2,
    'QM2_Val_POD03_Power': 7.8,
    'QM2_Val_POD04_Power': 8,
    'QM2_NAV_STW_Longitudinal': 23.5,
    'QM2_Ship_Outside_Temperature': 16.8,
    'Distance_nautical_miles': 22.78,
    'Departure_Arrival':0,
    'Year':2023

}

# Initialize the total fuel consumption variable
Voyagea_Fuel_Consumption = 0

# Loop n times
for _ in range(152):
    predicted_consumption = predict_rnn(input_features)
    Voyagea_Fuel_Consumption += predicted_consumption

print(f"Voyage Total Fuel Consumption: {Voyagea_Fuel_Consumption}")

# Plotting actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Fuel Consumption')
plt.plot(np.arange(y_test.min(), y_test.max()), np.arange(y_test.min(), y_test.max()), 'r--')
plt.show()

# Plotting training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.show()

# Save the model architecture as an image
plot_model(model, to_file='rnn_model.png', show_shapes=True, show_layer_names=True)

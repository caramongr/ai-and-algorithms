import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

# Load your dataset and preprocess it as needed
data = pd.read_excel('SummaryDataFile.xlsx')

# Encode 'Departure_Arrival' column using one-hot encoding
data = pd.get_dummies(data, columns=['Departure_Arrival'], drop_first=True)

# Select features and target variable
X = data.drop(['Time_Started', 'Time_Ended'], axis=1)
y = data['Total_Fuel_Consumption']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test = scaler_y.transform(y_test.values.reshape(-1, 1))

# Reshape the data to match the input shape expected by an RNN
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Build the RNN model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), verbose=1)

# Make predictions
y_pred = model.predict(X_test)

# Inverse transform the scaled predictions to the original scale
y_pred_original_scale = scaler_y.inverse_transform(y_pred).flatten()
y_test_original_scale = scaler_y.inverse_transform(y_test).flatten()

# Calculate RMSE
rmse = np.sqrt(np.mean((y_pred_original_scale - y_test_original_scale) ** 2))
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Plot the model's training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()



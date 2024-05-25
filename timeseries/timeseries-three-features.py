import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

# Simulated weather data
data = {
    'Temperature': [30.5, 32.0, 31.8, 33.2, 34.1, 33.5, 35.0, 34.8, 36.2, 37.1, 36.5, 37.0, 38.1, 39.2, 38.5, 37.8, 39.0, 40.1, 39.5, 40.2,
                    41.0, 42.2, 43.1, 44.0, 43.5, 42.8, 42.0, 41.5, 40.8, 40.0, 39.8, 39.5, 39.2, 39.0, 38.5, 38.0, 37.5, 37.0, 36.5, 36.0],
    'Humidity': [65, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65,
                 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45],
    'Wind Speed': [5, 5.5, 5.8, 6, 6.2, 6.5, 6.8, 7, 7.2, 7.5, 7.8, 8, 8.2, 8.5, 8.8, 9, 9.2, 9.5, 9.8, 10,
                   10.2, 10.5, 10.8, 11, 11.2, 11.5, 11.8, 12, 12.2, 12.5, 12.8, 13, 13.2, 13.5, 13.8, 14, 14.2, 14.5, 14.8, 15]
}
df = pd.DataFrame(data)

# Visualize the data
df.plot(subplots=True, figsize=(10, 6))
plt.show()

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Prepare the data for LSTM
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])  # Predicting temperature
    return np.array(X), np.array(y)

seq_length = 10
X, y = create_sequences(scaled_data, seq_length)

# Split the data into training and testing sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Reshape the data to fit the LSTM input shape
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, batch_size=1)

# Plot training & validation loss values
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Predict the temperatures
y_pred = model.predict(X_test)

# Inverse transform the predictions and actual values
y_test_inv = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 2)))))
y_pred_inv = scaler.inverse_transform(np.hstack((y_pred, np.zeros((y_pred.shape[0], 2)))))

# Extract the temperature predictions and actual values
y_test_inv = y_test_inv[:, 0]
y_pred_inv = y_pred_inv[:, 0]

# Plot predictions vs actual values
plt.figure(figsize=(8,4))
plt.plot(y_test_inv, label='Actual Temperature')
plt.plot(y_pred_inv, label='Predicted Temperature')
plt.title('Temperature Prediction')
plt.xlabel('Time Step')
plt.ylabel('Temperature')
plt.legend()
plt.show()

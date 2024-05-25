import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

# Load the dataset
# For this example, we'll simulate some temperature data
# In practice, you would load real data from a file like this:
# df = pd.read_csv('weather.csv')

# Simulated weather data (temperature)
data = {
    'Temperature': [30.5, 32.0, 31.8, 33.2, 34.1, 33.5, 35.0, 34.8, 36.2, 37.1, 36.5, 37.0, 38.1, 39.2, 38.5, 37.8, 39.0, 40.1, 39.5, 40.2]
}

# print(data)
# print(type(data))

df = pd.DataFrame(data)
# print(df)

# Visualize the data
plt.plot(df['Temperature'])
plt.title('Temperature Data')
plt.xlabel('Time Step')
plt.ylabel('Temperature')
plt.show()

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# print(scaled_data)

# Prepare the data for LSTM
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 5
X, y = create_sequences(scaled_data, seq_length)
# print(X)
# print("----")
# print(y)

# Split the data into training and testing sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# print("X_train")
# print(X_train)

# Reshape the data to fit the LSTM input shape
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# print("X_train")
# print(X_train)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=20, validation_split=0.2, batch_size=1)

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
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_inv = scaler.inverse_transform(y_pred)

# Plot predictions vs actual values
plt.figure(figsize=(8,4))
plt.plot(y_test_inv, label='Actual Temperature')
plt.plot(y_pred_inv, label='Predicted Temperature')
plt.title('Temperature Prediction')
plt.xlabel('Time Step')
plt.ylabel('Temperature')
plt.legend()
plt.show()

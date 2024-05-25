import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Generate synthetic dataset
np.random.seed(0)
X_train = np.random.rand(1000, 20)  # 1000 samples, each with 20 features
y_train = np.random.randint(2, size=1000)  # Binary labels (0 or 1)

# Define the Sequential model
model = Sequential()

# Add layers to the model
model.add(Dense(64, activation='relu', input_shape=(20,)))  # Input layer with 20 input neurons
model.add(Dense(64, activation='relu'))  # Hidden layer with 64 neurons
model.add(Dense(1, activation='sigmoid'))  # Output layer with 1 neuron (for binary classification) and sigmoid activation

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_train, y_train)
print("Training Accuracy:", accuracy)
print(model.summary())

import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model, to_categorical

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)

sns.set(font_scale=2)
import numpy as np

index = np.random.choice(np.arange(len(X_train)), 24, replace=False)
figure, axes = plt.subplots(nrows=4, ncols=6, figsize=(16, 9))
for item in zip(axes.ravel(), X_train[index], y_train[index]):
    axes, image, target = item
    axes.imshow(image, cmap=plt.cm.gray_r)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(target)

plt.tight_layout()

# plt.show()

X_train = X_train.reshape(60000, 28, 28, 1)

X_train = X_train.astype('float32') / 255

X_test = X_test.reshape(10000, 28, 28, 1)

X_test = X_test.astype('float32') / 255

y_train = to_categorical(y_train)

y_test = to_categorical(y_test)

cnn = Sequential()

cnn.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', 
               input_shape=(28, 28, 1))) 
cnn.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu')) 


cnn.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu')) 
cnn.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu')) 


cnn.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu')) 
cnn.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu')) 


cnn.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu')) 
cnn.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu')) 

cnn.add(Flatten())
cnn.add(Dense(units=128, activation='relu'))
cnn.add(Dense(units=10, activation='softmax'))


cnn.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

history = cnn.fit(X_train, 
        y_train, 
        epochs=2, 
        batch_size=64,
        validation_split=0.1, 
        verbose=0)

loss, accuracy = cnn.evaluate(X_test, y_test)

print(f'\nAccuracy: {accuracy:3%}')
print(f'loss: {loss:3%}')


# weights_first_layer = cnn.layers[0].get_weights()

# Access the weights of the second layer
# weights_second_layer = cnn.layers[2].get_weights()

# Access the weights of any other layer in a similar fashion

# You can print and analyze the weights
# print("Weights of the first layer:")
# print(weights_first_layer)

# print("\nWeights of the second layer:")
# print(weights_second_layer)

# plot_model(cnn, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

predictions = cnn.predict(X_test)
# print(predictions[0])
print(y_test[0])


for index, probability in enumerate(predictions[0]):
    print(f'{index}: {probability:.10%}')

incorrect_predictions = []

for i, (predicted, actual) in enumerate(zip(predictions, y_test)):
    predicted, exprcted = np.argmax(predicted), np.argmax(actual)  
    if predicted != exprcted:
        incorrect_predictions.append((i, predicted, exprcted))

print(len(incorrect_predictions))

def display_probabilities(predictions):
    for index, probability in enumerate(predictions):
        print(f'{index}: {probability:.10%}')

display_probabilities(predictions[495])
display_probabilities(predictions[652])

cnn.save('mnist_cnn.h5')




import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

# Load the pre-trained InceptionV3 model
base_model = InceptionV3(weights='imagenet', include_top=False)


# Print all layer names and their index in the model
for i, layer in enumerate(base_model.layers):
    print(i, layer.name, layer.__class__.__name__)

# Maximize the activations of these layers
names = ['mixed0','mixed3', 'mixed5']
layers = [base_model.get_layer(name).output for name in names]

# Create a model that returns the layer activations
dream_model = Model(inputs=base_model.input, outputs=layers)



def compute_loss(input_image):
    features = dream_model(input_image)
    loss = tf.zeros(shape=())
    for feature in features:
        # Normalize by number of elements in each feature map
        loss += tf.reduce_mean(tf.square(feature)) / tf.cast(tf.size(feature), tf.float32)
    return loss

@tf.function
def deepdream_step(img, step_size):
    with tf.GradientTape() as tape:
        # 'watch' the input image
        tape.watch(img)
        loss = compute_loss(img)
    
    # Calculate gradients of the loss with respect to the image
    gradients = tape.gradient(loss, img)
    
    # Normalize gradients
    gradients /= tf.math.reduce_std(gradients) + 1e-8 
    
    # Update the image by following the gradients
    img += gradients * step_size
    return img

def run_deep_dream_simple(img, steps=100, step_size=0.01):
    # Convert from uint8 to the range expected by the model
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    img = tf.convert_to_tensor(img)
    for step in range(steps):
        img = deepdream_step(img, step_size)
        
        if step % 10 == 0:
            print("Step {}, loss {}".format(step, compute_loss(img).numpy()))
    
    # Convert from model input range to uint8
    result = deprocess_image(img.numpy())
    return result

def load_image(filename):
    img = image.load_img(filename, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

def deprocess_image(img):
    img = 255 * (img + 1.0) / 2.0
    return np.clip(img, 0, 255).astype('uint8')

# Load and preprocess an image
original_img = load_image('me.jpg')  # Replace with your image path
dream_img = run_deep_dream_simple(original_img, steps=100, step_size=0.01)

plt.imshow(dream_img[0])
plt.show()

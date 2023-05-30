import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

def fgsm(model, image, epsilon):
    # Create a tensor for the input image
    input_image = tf.cast(image, tf.float32)
    input_image = tf.expand_dims(input_image, axis=0)

    # Create a tensor for the true label of the input image
    true_label = tf.argmax(model.predict(input_image), axis=1)

    # Calculate the gradient of the loss function with respect to the input image
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        loss = tf.keras.losses.categorical_crossentropy(model(input_image), tf.one_hot(true_label, depth=1000))
    gradient = tape.gradient(loss, input_image)

    # Create a perturbed image by adding a small epsilon value to the sign of the gradient
    perturbed_image = input_image + epsilon * tf.sign(gradient)
    perturbed_image = tf.clip_by_value(perturbed_image, 0, 255)

    # Return the perturbed image and the true label of the input image
    return perturbed_image.numpy(), true_label.numpy()


# Load the ResNet50 model
model = ResNet50(weights='imagenet')

# Load and preprocess the input image
img_path = '/home/shaheen/Downloads/panda.jpeg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Generate an adversarial example using the FGSM attack
epsilon = 30
perturbed_image, true_label = fgsm(model, img, epsilon)


# Decode the predictions for the original and perturbed images
preds_original = model.predict(x)
preds_perturbed = model.predict(perturbed_image)

# Get the index of the predicted class for the original and perturbed images
pred_class_original = np.argmax(preds_original, axis=1)[0]
pred_class_perturbed = np.argmax(preds_perturbed, axis=1)[0]

# Get the label associated with the predicted index for the original and perturbed images
label_original = decode_predictions(preds_original, top=1)[0][0][1]
label_perturbed = decode_predictions(preds_perturbed, top=1)[0][0][1]

# Print the label of the predicted class for the original and perturbed images
print('Original image prediction:', label_original)
print('Perturbed image prediction:', label_perturbed)

# Reshape the perturbed image to (224, 224, 3)
perturbed_image = np.squeeze(perturbed_image, axis=0)

perturbed_image = np.clip(perturbed_image, 0, 255)

# convert pixel values to integers
perturbed_image = np.rint(perturbed_image).astype(np.uint8)

# display the perturbed image
fig, ax = plt.subplots(1, 2)
ax[0].imshow(img)
ax[0].set_title('Original Image\nprediction: {}'.format(label_original))
ax[1].imshow(perturbed_image)
ax[1].set_title('Perturbed Image\nprediction: {}'.format(label_perturbed))
plt.show()



import os
import cv2
import PIL
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential, load_model, model_from_json

from google.colab import drive
drive.mount('/content/drive')
img1 =image.load_img('/content/Apple scab.jpg')
train_dir ='/content/drive/MyDrive/train'
test_dir = '/content/drive/MyDrive/valid'
IMG_SIZE = (256, 256)
BATCH_SIZE = 16

train_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.3
    )
val_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.3
    )

train_set = train_gen.flow_from_directory(
    '/content/drive/MyDrive/train',
    subset = 'training',
    class_mode = 'categorical',
    target_size = IMG_SIZE,
    batch_size = 8
)

val_set = val_gen.flow_from_directory(
  '/content/drive/MyDrive/valid',
   subset = 'validation',
   class_mode = 'categorical',
   target_size = IMG_SIZE,
   batch_size = 8
   )

train_set.classes

train_set.class_indices

model = tf.keras.models.Sequential([

        layers.InputLayer(input_shape=(256, 256, 3)),

        layers.Conv2D( 32, 3,padding='valid', activation='relu'),

        layers.MaxPooling2D(pool_size=(2,2)),
        #  #########

        layers.Conv2D( 64, 3,padding='valid', activation='relu'),

        layers.MaxPooling2D(pool_size=(2,2)),
        #  #########

        layers.Conv2D( 64, 3,padding='valid', activation='relu'),

        layers.MaxPooling2D(pool_size=(2,2)),
        #  #########

        layers.Conv2D( 64, 3,padding='valid', activation='relu'),

        layers.MaxPooling2D(pool_size=(2,2)),
        #  #########

        layers.Conv2D( 64, 3,padding='valid', activation='relu'),

        layers.MaxPooling2D(pool_size=(2,2)),
        # ##########

        layers.Flatten(),

        layers.Dense(64, activation='relu'),

        layers.Dense(4, activation='softmax')
        ])

print(model.summary())


model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=['accuracy']
    )
final_model = model.fit(
     train_set,
     epochs=3,
     validation_data=val_set,
     steps_per_epoch = len(train_set),
     validation_steps = len(val_set)
     )


test_gen = ImageDataGenerator(rescale=1./255)

test_set = test_gen.flow_from_directory(
    '/content/drive/MyDrive/valid',
    class_mode = 'categorical',
    target_size = IMG_SIZE,
    batch_size = 8
)



final_model.params

final_model.history.keys()

model.save('Apple_Disease_Detection.h5')

def predict(model, images):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i])
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = test_set.classes[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)

    return predicted_class, confidence

scores = model.evaluate(test_set, batch_size=32, verbose=2)


import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Load the trained model
model = keras.models.load_model('Apple_Disease_Detection.h5')

# Load the test dataset
test_dir = '/content/drive/MyDrive/valid'
IMG_SIZE = (256, 256)
BATCH_SIZE = 8

test_gen = ImageDataGenerator(rescale=1./255)
test_set = test_gen.flow_from_directory(
    test_dir,
    class_mode='categorical',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Evaluate the model
loss, accuracy = model.evaluate(test_set)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)

# Generate predictions
predictions = model.predict(test_set)
predicted_classes = np.argmax(predictions, axis=1)

# True labels
true_classes = test_set.classes

# Create confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
print(cm)

# Classification report
class_labels = test_set.class_indices.keys()
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)



import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import numpy as np




from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = Image.open('/content/drive/MyDrive/train/Cedarapplerust/CedarAppleRust(13).JPG')

# Resize the image while preserving the aspect ratio
max_dimension = 300  # Adjust this value as needed
image.thumbnail((max_dimension, max_dimension))

# Load your Sequential model from a file
model_path = './Apple_Disease_Detection.h5'
model = tf.keras.models.load_model(model_path)

# Create a new model with the same inputs as the loaded model
layer_outputs = [layer.output for layer in model.layers]
intermediate_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)

# Convert the image to an array and preprocess it
image_array = tf.keras.preprocessing.image.img_to_array(image)
image_array = tf.keras.applications.vgg16.preprocess_input(image_array)
image_array = tf.expand_dims(image_array, axis=0)

# Pass the image through the intermediate model to get the layer activations
activations = intermediate_model.predict(image_array)

# Iterate over each layer's activation and visualize all channels
for layer_index, layer_activation in enumerate(activations):
    # Determine the number of channels in the layer's activation
    channels = layer_activation.shape[-1]

    # Determine the number of rows and columns for subplots
    cols = min(channels, 8)  # Maximum 8 columns for better visibility
    rows = (channels - 1) // cols + 1

    # Create the figure and axes for subplots
    fig, ax = plt.subplots(rows, cols, figsize=(15, rows * 3))
    fig.suptitle(f'Layer {layer_index} - {model.layers[layer_index].name}', fontsize=14)

    # Iterate over channels and plot the activations
    for i in range(channels):
        # Get the activations of the current channel
        channel_activations = layer_activation[..., i]

        # Normalize activations to the range [0, 255]
        normalized_activations = ((channel_activations - np.min(channel_activations)) /
                                  (np.max(channel_activations) - np.min(channel_activations))) * 255

        # Convert to unsigned 8-bit integers
        normalized_activations = normalized_activations.astype(np.uint8)

        # Check the shape of normalized_activations
        if len(normalized_activations.shape) < 3:
            normalized_activations = np.expand_dims(normalized_activations, axis=-1)

        # Create a gray-scale image from the normalized activations
        gray_image = Image.fromarray(normalized_activations[0], mode='L')

        # Display the gray-scale image
        if rows == 1:
            ax[i % cols].imshow(gray_image, cmap='gray')
            ax[i % cols].axis('off')
            ax[i % cols].set_title(f'Channel {i+1}')
        else:
            ax[i // cols, i % cols].imshow(gray_image, cmap='gray')
            ax[i // cols, i % cols].axis('off')
            ax[i // cols, i % cols].set_title(f'Channel {i+1}')

    # Adjust layout and display the figure
    plt.tight_layout()
    plt.show()




# Iterate over each layer's activation and visualize up to 8 channels
for layer_index, layer_activation in enumerate(activations):
    # Determine the number of channels in the layer's activation
    channels = layer_activation.shape[-1]
    max_channels = min(channels, 8)  # Limit the number of channels to 8

    # Determine the number of rows and columns for subplots
    cols = min(max_channels, 8)  # Maximum 8 columns for better visibility
    rows = (max_channels - 1) // cols + 1

    # Create the figure and axes for subplots
    fig, ax = plt.subplots(rows, cols, figsize=(15, rows * 3))
    fig.suptitle(f'Layer {layer_index} - {model.layers[layer_index].name}', fontsize=14)

    # Iterate over channels and plot the activations
    for i in range(max_channels):
        # Get the activations of the current channel
        channel_activations = layer_activation[..., i]

        # Normalize activations to the range [0, 255]
        normalized_activations = ((channel_activations - np.min(channel_activations)) /
                                  (np.max(channel_activations) - np.min(channel_activations))) * 255

        # Convert to unsigned 8-bit integers
        normalized_activations = normalized_activations.astype(np.uint8)

        # Check the shape of normalized_activations
        if len(normalized_activations.shape) < 3:
            normalized_activations = np.expand_dims(normalized_activations, axis=-1)

        # Create a gray-scale image from the normalized activations
        gray_image = Image.fromarray(normalized_activations[0], mode='L')

        # Display the gray-scale image
        if rows == 1:
            ax[i % cols].imshow(gray_image, cmap='gray')
            ax[i % cols].axis('off')
            ax[i % cols].set_title(f'Channel {i+1}')
        else:
            ax[i // cols, i % cols].imshow(gray_image, cmap='gray')
            ax[i // cols, i % cols].axis('off')
            ax[i // cols, i % cols].set_title(f'Channel {i+1}')

    # Remove any empty subplots
    if max_channels < rows * cols:
        if rows == 1:
            for j in range(max_channels, cols):
                fig.delaxes(ax[j])
        else:
            for j in range(max_channels, rows * cols):
                fig.delaxes(ax[j // cols, j % cols])

    # Adjust layout and display the figure
    plt.tight_layout()
    plt.show()


 # Iterate over channels and plot the activations
for i in range(channels):
        # Get the activations of the current channel
        channel_activations = layer_activation[..., i]

        # Normalize activations to the range [0, 255]
        normalized_activations = ((channel_activations - np.min(channel_activations)) /
                                  (np.max(channel_activations) - np.min(channel_activations))) * 255

        # Convert to unsigned 8-bit integers
        normalized_activations = normalized_activations.astype(np.uint8)

        # Check the shape of normalized_activations
        if len(normalized_activations.shape) < 3:
            normalized_activations = np.expand_dims(normalized_activations, axis=-1)

        # Create a gray-scale image from the normalized activations
        gray_image = Image.fromarray(normalized_activations[0], mode='L')

        # Display the gray-scale image
        if rows == 1:
            ax[i % cols].imshow(gray_image, cmap='gray')
            ax[i % cols].axis('off')
            ax[i % cols].set_title(f'Channel {i+1}')
        else:
            ax[i // cols, i % cols].imshow(gray_image, cmap='gray')
            ax[i // cols, i % cols].axis('off')
            ax[i // cols, i % cols].set_title(f'Channel {i+1}')

    # Adjust layout and display the figure
plt.tight_layout()
plt.show()



import os
import cv2
import PIL
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential, load_model, model_from_json


# Define a function to predict the disease and print the image with its class, reasons, features, and protection measures
def predict_disease(model, train_set, IMG_SIZE, image_path):
    img = load_img('/content/drive/MyDrive/train/Cedarapplerust/CedarAppleRust(13).JPG', target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image
    # Make the prediction
    predictions = model.predict(image_array)
    predicted_class_index = np.argmax(predictions[0])
    # Get the class labels from train_set
    class_labels = list(train_set.class_indices.keys())

    # Define reasons and protection measures based on predicted classes
    disease_reasons = {
        'AppleBlackrot': 'Apple black rot is a fungal disease that affects apple trees. It causes black, rotting lesions on the fruit, leaves, and branches. The fungus survives in fallen leaves and mummified fruit during the winter and spreads through rain and wind. Infected fruits should be removed and destroyed to prevent further spread of the disease. Prune the affected branches during the dormant season, and apply fungicides during the growing season to protect new growth.',
        'AppleHealthy': 'No disease detected. The apple tree is healthy.',
        'AppleScab': 'Apple scab is a fungal disease that affects apple trees. It causes dark, scaly lesions on the leaves, fruit, and twigs. The fungus survives in infected fallen leaves and spreads through rain and wind. To prevent apple scab, it is important to maintain good sanitation by removing and destroying infected leaves and fruit. Fungicide sprays can also be used as a preventive measure. Choose apple cultivars that are resistant to scab.',
        'Cedarapplerust': 'Cedar apple rust is a fungal disease that affects apple trees. It causes orange or rust-colored spots on the leaves, fruit, and twigs. The disease spreads through spores that are released by cedar trees during wet weather. To prevent cedar apple rust, it is important to remove nearby cedar trees, as they serve as a host for the disease. Fungicide sprays can also be used to control the disease. Plant apple varieties that are resistant to cedar apple rust.'
    }

    protection_measures = {
        'AppleBlackrot': '1. Remove and destroy infected fruits, leaves, and branches.\n2. Prune affected branches during the dormant season.\n3. Apply fungicides during the growing season to protect new growth.\n4. Improve air circulation around the tree by pruning and thinning branches.\n5. Avoid overhead irrigation to reduce leaf wetness.',
        'AppleHealthy': 'No protection measures required.',
        'AppleScab': '1. Remove and destroy infected leaves and fruit.\n2. Maintain good sanitation by removing fallen leaves.\n3. Apply fungicides as a preventive measure.\n4. Choose apple cultivars that are resistant to scab.',
        'Cedarapplerust': '1. Remove nearby cedar trees, as they serve as a host for the disease.\n2. Apply fungicides to control the disease.\n3. Plant apple varieties that are resistant to cedar apple rust.\n4. Maintain good sanitation by removing fallen leaves and fruit.'
    }

    # Get the predicted class and confidence
    predicted_class = class_labels[predicted_class_index]
    confidence = round(100 * np.max(predictions[0]), 2)

    # Get the top 3 predicted classes and their corresponding probabilities
    top_3_indices = np.argsort(predictions[0])[::-1][:3]
    top_3_classes = [class_labels[i] for i in top_3_indices]
    top_3_probabilities = predictions[0][top_3_indices]
    # Print the prediction result
    print("Predicted Class:", predicted_class)
    print("Confidence:", confidence)
    print("Top 3 Predictions:")
    for i in range(3):
        print(f"{top_3_classes[i]}: {round(100 * top_3_probabilities[i], 2)}%")

    # Print the reasons for the predicted disease
    print("Reasons:")
    print(disease_reasons.get(predicted_class, "Reasons not available."))

    # Display the image
    plt.imshow(img)
    plt.axis('off')
    plt.title("Predicted Class: " + predicted_class + "\nConfidence: " + str(confidence) + "%")
    plt.show()

    # Get the features that contributed to the prediction
    intermediate_layer_model = keras.Model(inputs=model.input, outputs=model.layers[-2].output)
    features = intermediate_layer_model.predict(img_array)

    # Print the features
    print("Features:")
    print(features)

    # Print the protection measures for the predicted disease
    print("Protection Measures:")
    print(protection_measures.get(predicted_class, "Protection measures not available."))

# Usage example
model_path = 'Apple_Disease_Detection.h5'
train_dir = '/content/drive/MyDrive/train'
IMG_SIZE = (256, 256)
image_path = '/content/drive/MyDrive/train/Cedarapplerust/CedarAppleRust(13).JPG'

model = load_model(model_path)
train_gen = ImageDataGenerator(rescale=1./255, validation_split=0.3)
train_set = train_gen.flow_from_directory(
    train_dir,
    subset='training',
    class_mode='categorical',
    target_size=IMG_SIZE,
    batch_size=8
)

predict_disease(model, train_set, IMG_SIZE, image_path)

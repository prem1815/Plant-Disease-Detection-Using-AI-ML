import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random
from sklearn import preprocessing
import keras as keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import warnings
warnings.filterwarnings("ignore")

SIZE = 256
CHANNELS = 3
n_classes = 3
EPOCHS = 50
BATCH_SIZE = 8
input_shape = (SIZE, SIZE, CHANNELS)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.2,
    zoom_range=0.2,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory=r'C:\Users\smoha\OneDrive\Desktop\Notes\DPSD\DPSD PROJECT\Potato Leaf Dataset\train',
    target_size=(256, 256),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode="rgb",
    shuffle=True,
    seed=65
)

validation_generator = validation_datagen.flow_from_directory(
    r'C:\Users\smoha\OneDrive\Desktop\Notes\DPSD\DPSD PROJECT\Potato Leaf Dataset\valid',
    target_size=(256, 256),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode="rgb",
    shuffle=True,
    seed=76
)

test_generator = test_datagen.flow_from_directory(
    r'C:\Users\smoha\OneDrive\Desktop\Notes\DPSD\DPSD PROJECT\Potato Leaf Dataset\test',
    target_size=(256, 256),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode="rgb",
    shuffle=False,
    seed=42
)

model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.5),

    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.5),

    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Flatten(),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(n_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=['accuracy']
)

print(train_generator.n)
print(validation_generator.n)
print(train_generator.batch_size)
print(validation_generator.batch_size)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

score = model.evaluate_generator(test_generator)
print('Test loss : ', score[0])
print('Test accuracy : ', score[1])

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), acc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# Function to predict the class of an image and suggest fertilizer amount with percentage
def predict_and_suggest_fertilizer(image_path, model):
    try:
        img = cv2.imread(image_path)

        if img is None:
            raise Exception("Image not loaded properly.")

        img = cv2.resize(img, (SIZE, SIZE))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        predictions = model.predict(img)
        class_labels = ['Low Nitrogen', 'Balanced', 'High Nitrogen']
        class_probabilities = dict(zip(class_labels, predictions.flatten()))
        sorted_probabilities = sorted(class_probabilities.items(), key=lambda x: x[1], reverse=True)

        top_class, top_probability = sorted_probabilities[0]

        if top_class == 'Low Nitrogen':
            fertilizer_suggestion = "Use a Low Nitrogen Fertilizer"
        elif top_class == 'Balanced':
            fertilizer_suggestion = "Use a Balanced Fertilizer"
        elif top_class == 'High Nitrogen':
            fertilizer_suggestion = "Use a High Nitrogen Fertilizer"
        else:
            fertilizer_suggestion = "Unable to suggest fertilizer"

        # Calculate percentage based on the top probability
        fertilizer_percentage = round(top_probability * 100, 2)

        return fertilizer_suggestion, top_class, fertilizer_percentage

    except Exception as e:
        print(f"Error: {str(e)}")
        return "Error", None, None

# Get user input for image path
user_image_path = input("Enter the path of the image you want to check: ")

# Example usage
suggested_fertilizer, top_class, fertilizer_percentage = predict_and_suggest_fertilizer(user_image_path, model)

if top_class is not None and fertilizer_percentage is not None:
    print('Predicted Class:', top_class)
    print('Approximate Fertilizer Percentage:', fertilizer_percentage, '%')
    print('Suggested Fertilizer:', suggested_fertilizer)
else:
    print("Exiting the program.")

model.save('spray_model.h5')

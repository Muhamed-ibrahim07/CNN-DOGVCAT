import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Define paths
train_dir = "dataset/train"
test_dir = "dataset/test"

# ImageDataGenerator for loading and augmenting images
train_datagen = ImageDataGenerator(rescale=1.0/255.0, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Load the training and testing data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),  # Resize images to 128x128
    batch_size=32,
    class_mode="binary"  # Binary classification for cats vs dogs
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode="binary"
)

# Define the CNN model
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),  # 3 channels for RGB
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation="relu"),
    layers.Dense(1, activation="sigmoid")  # Binary classification: 1 output neuron with sigmoid
])

# Compile the model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
model.fit(train_generator, epochs=5, validation_data=test_generator)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_generator)
print("Test accuracy:", test_acc)

# Make predictions on a test image
image_path = "test_image_dog_or_cat.jpg"  # Path to a custom test image
image = Image.open(image_path).convert("RGB").resize((128, 128))
image_arr = np.array(image).astype("float32") / 255.0
image_arr = np.expand_dims(image_arr, 0)  # Add batch dimension

# Predict whether it's a cat or dog
prediction = model.predict(image_arr)
predicted_label = "dog" if prediction[0] > 0.5 else "cat"
print("Predicted label:", predicted_label)


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image

def load_and_preprocess_images(image_paths, image_size=(28, 28)):
    images = []
    for path in image_paths:
        img = Image.open(path).convert('L')  # Convert to grayscale
        img = img.resize(image_size)
        img = np.array(img)
        img = img / 255.0  # Normalize images to [0, 1]
        images.append(img)
    return np.array(images)

def build_autoencoder(input_shape=(28, 28, 1)):
    # Encoder
    encoder = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)
    ])

    # Decoder
    decoder = models.Sequential([
        layers.Conv2DTranspose(64, kernel_size=3, strides=2, activation='relu', padding='same'),
        layers.Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu', padding='same'),
        layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')
    ])

    autoencoder = models.Sequential([encoder, decoder])
    return autoencoder


def train_autoencoder_for_character(character_images, epochs=50):
    autoencoder = build_autoencoder()
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    character_images = np.expand_dims(character_images, -1)  # Add channel dimension if not already present
    autoencoder.fit(character_images, character_images, epochs=epochs, batch_size=32)
    return autoencoder

def train_autoencoders_for_dataset(character_dataset_path):
    trained_autoencoders = {}
    for character in os.listdir(character_dataset_path):
        char_path = os.path.join(character_dataset_path, character)
        if os.path.isdir(char_path):
            image_paths = [os.path.join(char_path, f) for f in os.listdir(char_path) if f.endswith('.png')]
            images = load_and_preprocess_images(image_paths)
            autoencoder = train_autoencoder_for_character(images)
            trained_autoencoders[character] = autoencoder
    return trained_autoencoders

# Example usage:
# character_dataset_path = 'path_to_your_dataset'
# trained_autoencoders = train_autoencoders_for_dataset(character_dataset_path)

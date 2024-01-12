import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from PIL import Image

def load_and_preprocess_images(image_paths, image_size=(32, 32)):
    images = []
    for path in image_paths:
        img = Image.open(path).convert('L')  # convert to grayscale
        img = img.resize(image_size)
        img = np.array(img)
        img = (img - 127.5) / 127.5  # Normalize images to [-1, 1]
        images.append(img)
    return np.array(images)

def build_generator_model(output_shape=(28, 28, 1)):
    model = models.Sequential([
        # Starting with a Dense layer that maps the input to a suitable number of units
        layers.Dense(8 * 8 * 256, use_bias=False, input_shape=(100,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((8, 8, 256)),
        
        # Upsampling to the target image size
        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        # Final layer with dimensions matching the training images
        layers.Conv2DTranspose(output_shape[-1], (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model

def build_discriminator_model():
    model = models.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[32, 32, 1]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

def get_dimensions(character, character_dataset_path):
    folder_path = os.path.join(character_dataset_path, character)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        with Image.open(file_path) as img:
            return img.size  # Return the dimensions of the first image

    return 28, 28  # Return a default value if no images are found


def train_gan_for_character(character, character_images, character_dataset_path, epochs = 50):
    avg_width, avg_height = get_dimensions(character, character_dataset_path)
    generator = build_generator_model(output_shape=(avg_height, avg_width, 1))
    discriminator = build_discriminator_model()

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    generator_optimizer = optimizers.Adam(1e-4)
    discriminator_optimizer = optimizers.Adam(1e-4)

    @tf.function
    def train_step(images):
        noise = tf.random.normal([len(images), 100])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)
            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
            disc_loss = cross_entropy(tf.ones_like(real_output), real_output) + cross_entropy(tf.zeros_like(fake_output), fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    for epoch in range(epochs):
        for image_batch in tf.data.Dataset.from_tensor_slices(character_images).batch(32):
            train_step(image_batch)

    return generator

def train_gans_for_dataset(character_dataset_path, models_save_path='trained_gan_models'):
    os.makedirs(models_save_path, exist_ok=True)  # Create the models directory if it doesn't exist

    for character in os.listdir(character_dataset_path):
        char_path = os.path.join(character_dataset_path, character)
        if os.path.isdir(char_path):
            print(character)
            image_paths = [os.path.join(char_path, f) for f in os.listdir(char_path) if f.endswith('.png')]
            images = load_and_preprocess_images(image_paths)
            images = np.expand_dims(images, -1)  # Add channel dimension

            # Train GAN for this character
            generator = train_gan_for_character(character, images, character_dataset_path)

            # Save the generator model
            generator_save_path = os.path.join(models_save_path, f'{character}_generator.h5')
            generator.save(generator_save_path)

# Example usage:
character_dataset_path = 'path_to_save_cropped_images'
train_gans_for_dataset(character_dataset_path)
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os

def load_model_for_character(character, models_save_path='trained_gan_models'):
    model_path = os.path.join(models_save_path, f'{character}_generator.h5')
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        print(f"Model for {character} not found.")
        return None

# Example of loading a model:
# loaded_model = load_model_for_character('a')

def generate_image(generator_model, noise_dim=100):
    # Generate random noise (the input for the generator)
    random_noise = tf.random.normal([1, noise_dim])

    # Generate an image from the noise
    generated_image = generator_model(random_noise, training=False)

    # Post-process the generated image
    generated_image = (generated_image * 127.5 + 127.5).numpy()  # Rescale the image values
    generated_image = np.squeeze(generated_image).astype(np.uint8)  # Remove batch dimension and convert to uint8

    return Image.fromarray(generated_image)

# Assuming 'my_generator_model' is your loaded or defined generator model
# generated_image = generate_image(my_generator_model)

# To display or save the image:
# generated_image.show()  # To display the image
# generated_image.save('generated_image.png')  # To save the image


for c in 'abcdefghijklmnopqrstuvwxyz':
    print("CURRENT LETTER: ", c)
    model = load_model_for_character(c)
    if not model: continue
    generated_image = generate_image(model)
    if generated_image: generated_image.show()


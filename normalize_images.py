from PIL import Image
import os

def resize_image_and_fill(img, target_size=(32, 32)):
    # Calculate the ratio to resize the image to fit within the target size
    ratio = min(target_size[0] / img.width, target_size[1] / img.height)
    new_size = (max(int(img.width * ratio), 1), max(int(img.height * ratio), 1))  # Ensure at least 1 pixel

    # Resize the image using the LANCZOS filter
    img = img.resize(new_size, Image.Resampling.LANCZOS)

    # Create a new image with a white background
    new_img = Image.new("RGB", target_size, (255, 255, 255))
    # Paste the resized image onto the center of the new image
    new_img.paste(img, ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2))

    return new_img


def resize_images_to_fixed_size(cropped_images_dir, target_size=(32, 32)):
    for char_folder in os.listdir(cropped_images_dir):
        folder_path = os.path.join(cropped_images_dir, char_folder)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                with Image.open(file_path) as img:
                    resized_img = resize_image_and_fill(img, target_size)
                    resized_img.save(file_path)

# Example usage:
cropped_images_dir = 'path_to_save_cropped_images'
resize_images_to_fixed_size(cropped_images_dir)

import os
import cv2
import pytesseract
from PIL import Image
from collections import defaultdict
from datetime import datetime

def process_image_for_ocr(filepath):
    image = cv2.imread(filepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary)

def crop_characters_from_image(image, boxes):
    for b in boxes.splitlines():
        b = b.split(' ')
        char = b[0].lower()
        if len(char) == 1 and char.isalpha():
            x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])

            # Ensure the coordinates are within the image boundaries
            x, y, w, h = max(0, x), max(0, y), min(image.width, w), min(image.height, h)
            if w > x and h > y:  # Check if the coordinates are valid
                cropped_image = image.crop((x, image.height - h, w, image.height - y))
                yield char, cropped_image

def process_handwriting_samples(handwriting_samples_dir, cropped_images_dir):
    character_dataset = defaultdict(list)

    os.makedirs(cropped_images_dir, exist_ok=True)
    for filename in os.listdir(handwriting_samples_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')):
            filepath = os.path.join(handwriting_samples_dir, filename)
            processed_image = process_image_for_ocr(filepath)
            boxes = pytesseract.image_to_boxes(processed_image)

            for char, cropped_image in crop_characters_from_image(processed_image, boxes):
                char_folder = os.path.join(cropped_images_dir, char)
                os.makedirs(char_folder, exist_ok=True)
                unique_filename = f"{char}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.png"
                cropped_image_path = os.path.join(char_folder, unique_filename)
                cropped_image.save(cropped_image_path)
                character_dataset[char].append(cropped_image_path)

    return character_dataset



handwriting_samples_dir = 'path_to_handwriting_samples'
cropped_images_dir = 'path_to_save_cropped_images'
dataset = process_handwriting_samples(handwriting_samples_dir, cropped_images_dir)
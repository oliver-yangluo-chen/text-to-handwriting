# Revised script to copy images instead of moving them, and removing unnecessary variables

import pandas as pd
import os
import shutil

# Define file paths
csv_file_path = 'english.csv'
save_directory = 'path_to_save_cropped_images/'  # Directory to save the copied images

# Read the CSV file
data = pd.read_csv(csv_file_path)

# Filter out only lowercase and uppercase letters
filtered_data = data[data['label'].str.match('^[a-zA-Z]$', na=False)]

# Create the save directory if it doesn't exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Function to copy images to their corresponding folders
def copy_images(dataframe, save_dir):
    for index, row in dataframe.iterrows():
        label = row['label']
        image_path = row['image']  # Full path to the image
        target_folder = os.path.join(save_dir, label)

        # Create a folder for the label if it doesn't exist
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        # Copy the image to the corresponding folder if it exists
        if os.path.exists(image_path):
            shutil.copy(image_path, os.path.join(target_folder, os.path.basename(image_path)))
        else:
            print(f"Image not found: {image_path}")

# Call the function to copy the images
copy_images(filtered_data, save_directory)

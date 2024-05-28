import os
from PIL import Image
from config import *

def check_images(directory):
    print(f"Checking images in {directory}...")
    num_corrupted = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    img = Image.open(os.path.join(root, file))
                    img.verify()
                except (IOError, SyntaxError) as e:
                    print('Bad file:', os.path.join(root, file))
                    num_corrupted += 1
    print(f"Found {num_corrupted} corrupted images in {directory}.")

def check_all_images():
    check_images(TRAIN_DIR)
    check_images(VALIDATION_DIR)
    check_images(TEST_DIR)

if __name__ == '__main__':
    check_all_images()

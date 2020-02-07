# Importing packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

# Environment variables
TEST_IMAGES_DIRECTORY = 'test_images/'
TEST_IMAGES_OUTPUT_DIRECTORY = 'output_images/'
CAMERA_CALIBRATION_IMAGES = 'camera_cal/'
DEBUGGING_MODE = True  # False to suppress debugging output


# Function to test pipeline on test images
def test_sample_images(test_image_dir, output_image_dir):
    # Get test images
    image_paths = os.listdir(TEST_IMAGES_DIRECTORY)
    if not os.path.exists(TEST_IMAGES_OUTPUT_DIRECTORY):
        os.makedirs(TEST_IMAGES_OUTPUT_DIRECTORY)

    # Generate matplotlib figs for display purposes
    fig, axes = plt.subplots(len(image_paths), 2, figsize=(30, 30))
    axes = axes.flatten()
    ax_ctr = 0

    for image_path in image_paths:
        # Read the image into memory
        image = mpimg.imread(f'{TEST_IMAGES_DIRECTORY}{image_path}')

        # Show the original image
        ax = axes[ax_ctr]
        ax_ctr += 1
        ax.imshow(image)
        ax.title.set_text(f'Orignal: {image_path}')

        # Run image through pipeline

        # Show the resultant image
        ax = axes[ax_ctr]
        ax_ctr += 1
        ax.imshow(image)
        ax.title.set_text(f'Result: {image_path}')

        # Save the resultant image in the test directory
        cv2.imwrite(f'{TEST_IMAGES_OUTPUT_DIRECTORY}{image_path}', image)

    # Show final plot
    plt.tight_layout()
    plt.show()


def test_sample_videos():



# START OF MAIN PROGRAM
test_sample_images(TEST_IMAGES_DIRECTORY, TEST_IMAGES_OUTPUT_DIRECTORY)


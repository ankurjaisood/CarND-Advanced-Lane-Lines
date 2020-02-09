# Importing packages
import matplotlib.pyplot as plt
import cv2
import os
from moviepy.editor import VideoFileClip
from LaneDetection import find_lanes, calibrate_camera

# Environment variables
TEST_IMAGES_DIRECTORY = 'test_images/'
TEST_VIDEOS_DIRECTORY = 'test_videos/'
TEST_IMAGES_OUTPUT_DIRECTORY = 'output_images/'
TEST_VIDEOS_OUTPUT_DIRECTORY = 'output_videos/'
CAMERA_CALIBRATION_IMAGES = 'camera_cal/'
DEBUGGING_MODE = False  # False to suppress debugging output


# Function to test pipeline on test images
def test_sample_images(test_image_dir, output_image_dir, calibration_matrix, calibration_dist, debug_mode=False):
    # Get test images
    image_paths = os.listdir(test_image_dir)
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    # Generate matplotlib figs for display purposes
    fig, axes = plt.subplots(len(image_paths), 2, figsize=(30, 30))
    axes = axes.flatten()
    ax_ctr = 0

    for image_path in image_paths:
        # Read the image into memory
        image = cv2.imread(f'{test_image_dir}{image_path}')

        # Show the original image
        ax = axes[ax_ctr]
        ax_ctr += 1
        ax.imshow(image)
        ax.title.set_text(f'Orignal: {image_path}')

        # Run image through pipeline
        res = find_lanes(image, calibration_matrix, calibration_dist, debug_mode)

        # Show the resultant image
        ax = axes[ax_ctr]
        ax_ctr += 1
        ax.imshow(res)
        ax.title.set_text(f'Result: {image_path}')

        # Save the resultant image in the test directory
        cv2.imwrite(f'{output_image_dir}{image_path}', res)

    # Show final plot
    plt.tight_layout()
    plt.show()


def test_sample_videos(test_video_dir, output_video_dir):
    # Get test videos
    video_paths = os.listdir(test_video_dir)
    if not os.path.exists(output_video_dir):
        os.makedirs(output_video_dir)

    # Read each video, find lanes, write output video
    for video_path in video_paths:
        video = VideoFileClip(f'{test_video_dir}{video_path}')
        processed_video = video.fl_image(find_lanes)
        processed_video.write_videofile(f'{output_video_dir}{video_path}', audio=False)


# START OF MAIN PROGRAM

# Calibrate camera
success, cal_mtx, cal_dist = calibrate_camera(CAMERA_CALIBRATION_IMAGES, DEBUGGING_MODE)

test_sample_images(TEST_IMAGES_DIRECTORY, TEST_IMAGES_OUTPUT_DIRECTORY, cal_mtx, cal_dist, DEBUGGING_MODE)
#test_sample_videos(TEST_VIDEOS_DIRECTORY, TEST_VIDEOS_OUTPUT_DIRECTORY)

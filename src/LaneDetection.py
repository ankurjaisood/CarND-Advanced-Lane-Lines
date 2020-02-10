import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

'''
Algorithm Steps:
1. Camera Calibration
2. Distortion Correction
3. Colour and Gradient Threshold
4. Perspective Transform
5. Iterative Lane Detection Using Histogram and Polynomial Fitting
'''

# Camera calibration parameters
CHESSBOARD_SIZE = (9, 6)

# Thresholding parameters
GRADIENT_THRES_MIN = 40
GRADIENT_THRES_MAX = 100
COLOUR_THRES_MIN = 170
COLOUR_THRES_MAX = 255

# Perspective transform parameters
OFFSET = 100
SRC_POINTS = [[150, 700], [450, 500], [850, 500], [1200, 700]]

# Lane detection parameters



def calibrate_camera(calibration_images_dir, debug_mode=False):
    # Get all the image paths in the calibration images directory
    image_paths = os.listdir(calibration_images_dir)

    # Use the grid size to create the grid corner coordinates
    grid_points = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    grid_points[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

    # Create lists to store object points and image points
    object_points = []
    image_points = []

    # For each calibration image
    for i, image_path in enumerate(image_paths):
        # Read image and convert to grayscale
        image = np.array(cv2.imread(f'{calibration_images_dir}{image_path}'))
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        corners_found, corners = cv2.findChessboardCorners(gray_image, CHESSBOARD_SIZE, None)

        if corners_found:
            # Append the points to our list
            object_points.append(grid_points)
            image_points.append(corners)

            if debug_mode:
                # In debug mode show the images with points
                cv2.drawChessboardCorners(image, CHESSBOARD_SIZE, corners, corners_found)
                cv2.imshow(image_path, image)
                cv2.waitKey(500)
        elif not corners_found and debug_mode:
            print(f"Corners not found for image {i}: {image_path}")

    cv2.destroyAllWindows()

    if object_points and image_points:
        # Use the last image in the directory as the calibration image
        test_image = cv2.imread(f'{calibration_images_dir}{image_paths[-1]}')
        image_size = (test_image.shape[1], test_image.shape[0])

        # Calculate the calibration parameters
        success, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None, None)

        if debug_mode:
            # In debug mode show the result of the calibration
            corrected_image = cv2.undistort(test_image, mtx, dist, None, mtx)
            cv2.imshow("Original Image", test_image)
            cv2.imshow("Corrected Image", corrected_image)
            cv2.waitKey(500)

        return True, mtx, dist

    cv2.destroyAllWindows()
    return False, None, None


def perspective_transform(image, debug_mode=True):
    # Set source and destination points
    img_size = (image.shape[1], image.shape[0])
    src = np.float32(SRC_POINTS)
    dst = np.float32([[OFFSET, img_size[1] - OFFSET], [OFFSET, OFFSET], [img_size[0] - OFFSET, OFFSET], [img_size[0] - OFFSET, img_size[1] - OFFSET]])

    # Calculate perspective transform matrix
    perspective_matrix = cv2.getPerspectiveTransform(src, dst)

    # transform image
    transformed_image = cv2.warpPerspective(image, perspective_matrix, img_size)

    if debug_mode:
        cv2.polylines(image, np.array([SRC_POINTS], dtype=np.int32), False, 1, 3)

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.set_title('Original')
        ax1.imshow(image)

        ax2.set_title('Perspective Transform')
        ax2.imshow(transformed_image)

    return transformed_image


def threshold_image(image, debug_mode=False):
    # Convert to HLS colour space and extract s channel
    hls_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls_image[:, :, 2]

    # Get grayscale image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Sobel x
    abs_sobelx = np.absolute(cv2.Sobel(gray_image, cv2.CV_64F, 1, 0))
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold gradient and colour
    thres_sobel = np.zeros_like(scaled_sobel)
    thres_sobel[(scaled_sobel >= GRADIENT_THRES_MIN) & (scaled_sobel <= GRADIENT_THRES_MAX)] = 1
    thres_s_channel = np.zeros_like(s_channel)
    thres_s_channel[(s_channel >= COLOUR_THRES_MIN) & (s_channel <= COLOUR_THRES_MAX)] = 1

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(thres_sobel)
    combined_binary[(thres_sobel == 1) | (thres_s_channel == 1)] = 1

    if debug_mode:
        color_binary = np.dstack((np.zeros_like(thres_sobel), thres_sobel, thres_s_channel)) * 255

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.set_title('Stacked thresholds')
        ax1.imshow(color_binary)

        ax2.set_title('Combined S channel and gradient thresholds')
        ax2.imshow(combined_binary, cmap='gray')

    return combined_binary


def detect_and_fit_lines(image, debug_mode=False):
    # Create a histogram of the bottom part of the image
    histogram = np.sum(image[image.shape[0] // 2:, :], axis=0)

    if debug_mode:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.set_title('Perspective Image')
        ax1.imshow(image)

        ax2.set_title('Histogram')
        ax2.plot(histogram)

    # Sliding window

    return image


def find_lanes(image, mtx, dist, debug_mode=False):

    # Undistort using camera calibration
    undistorted_image = cv2.undistort(image, mtx, dist, None, mtx)

    # Colour and Gradient Thresholding
    thresholded_image = threshold_image(undistorted_image, debug_mode)

    # Perspective transform
    transformed_image = perspective_transform(thresholded_image, debug_mode)

    # Find lines
    detected_lines = detect_and_fit_lines(transformed_image, True)

    return detected_lines

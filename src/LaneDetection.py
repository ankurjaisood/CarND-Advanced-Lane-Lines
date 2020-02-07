import numpy as np
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

# CAMERA CALIBRATION PARAMETERS
CHESSBOARD_SIZE = (9, 6)


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


def perspective_transform(image):



def find_lanes(image, mtx, dist, debug_mode=False):

    # Undistort using camera calibration
    corrected_image = cv2.undistort(image, mtx, dist, None, mtx)

    # Perspective transform

    return image

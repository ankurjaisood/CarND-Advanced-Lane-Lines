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
GRADIENT_THRES_MIN = 30
GRADIENT_THRES_MAX = 100
COLOUR_THRES_MIN = 180
COLOUR_THRES_MAX = 255
MAGNITUDE_THRES_MIN = 80
MAGNITUDE_THRES_MAX = 180
DIR_THRES_MIN = 0.9
DIR_THRES_MAX = 1.3

# Perspective transform parameters
OFFSET = 100
SRC_POINTS = [[150, 700], [450, 500], [850, 500], [1200, 700]]

# Lane detection parameters
NUMBER_OF_WINDOWS = 10
WINDOW_MARGIN = 100
MIN_NUMBER_PIXELS = 200

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30 / 720  # meters per pixel in y dimension
xm_per_pix = 3.7 / 700  # meters per pixel in x dimension


# Define a class to receive the characteristics of each line detection
class Lane:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # polynomial coefficients for the most recent fit
        self.current_fit = None
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # current best position of x
        self.x_best = None
        # save all the detected x lane positions
        self.base = []
        # save all the detected fits
        self.fits = []
        # save last detection method used
        self.method = None
        # save lane center
        self.center = None

    def update_base(self, base):
        if self.x_best is None:
            self.base.append(base)
            self.x_best = base
        else:
            if abs(self.x_best - base) > 50:
                self.detected = False
            else:
                self.detected = True
                if len(self.base) >= 25:
                    self.base.pop(0)
                self.base.append(base)
                self.x_best = sum(self.base) / len(self.base)

    def update_fit(self, fit):
        if self.current_fit is None:
            self.current_fit = fit
            self.fits.append(fit)
        else:
            diff = self.current_fit / fit
            if diff[0] < 3 or diff[1] < 3 or diff[2] < 3:
                self.detected = False
            if len(self.fits) >= 10:
                self.fits.pop(0)
            self.fits.append(fit)
            self.current_fit = sum(self.fits) / len(self.fits)

    def update_center(self, width):
        center = np.absolute(width / 2 - self.x_best) * xm_per_pix
        self.center = center

    def radius_of_covergence(self):

        y_eval = 719

        # Calculation of R_curve (radius of curvature)
        radius = ((1 + (2 * self.current_fit[0] * y_eval * ym_per_pix + self.current_fit[1]) ** 2) ** 1.5) / np.absolute(2 * self.current_fit[0])

        if radius > 5000:
            radius = 5000
        if self.radius_of_curvature is None:
            self.radius_of_curvature = radius
        else:
            self.radius_of_curvature = self.radius_of_curvature*0.25 + radius*0.75


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


def get_perspective_transform(images_dir, debug_mode=False):

    # Read sample image
    image_paths = os.listdir(images_dir)
    image = cv2.imread(f'{images_dir}{image_paths[0]}')

    # Set source and destination points
    img_size = (image.shape[1], image.shape[0])
    src = np.float32(SRC_POINTS)
    dst = np.float32([[OFFSET, img_size[1] - OFFSET], [OFFSET, OFFSET], [img_size[0] - OFFSET, OFFSET], [img_size[0] - OFFSET, img_size[1] - OFFSET]])

    # Calculate perspective transform matrix
    perspective_matrix = cv2.getPerspectiveTransform(src, dst)
    inverse_perspective_matrix = cv2.getPerspectiveTransform(dst, src)

    return perspective_matrix, inverse_perspective_matrix


def perspective_transform(image, perspective_matrix, debug_mode=False):

    # transform image
    img_size = (image.shape[1], image.shape[0])
    transformed_image = cv2.warpPerspective(image, perspective_matrix, img_size)

    if debug_mode:
        cv2.polylines(image, np.array([SRC_POINTS], dtype=np.int32), False, 1, 3)

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.set_title('Original')
        ax1.imshow(image)

        ax2.set_title('Perspective Transform')
        ax2.imshow(transformed_image)
        cv2.waitKey(500)

    return transformed_image


def threshold_image(image, debug_mode=False):
    # Convert to HLS colour space and extract s channel
    hls_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls_image[:, :, 2]

    # Get grayscale image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Sobel x, y
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=9)
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=9)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    scaled_sobel_x = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    scaled_sobel_y = np.uint8(255 * abs_sobely / np.max(abs_sobely))

    # Sobel mag, dir
    gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    scaled_gradient_magnitude = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))

    gradient_dir = np.arctan2(abs_sobely, abs_sobelx)

    # Threshold gradient and colour
    thres_sobel_x = np.zeros_like(scaled_sobel_x)
    thres_sobel_x[(scaled_sobel_x >= GRADIENT_THRES_MIN) & (scaled_sobel_x <= GRADIENT_THRES_MAX)] = 1

    thres_sobel_y = np.zeros_like(scaled_sobel_y)
    thres_sobel_y[(scaled_sobel_y >= GRADIENT_THRES_MIN) & (scaled_sobel_y <= GRADIENT_THRES_MAX)] = 1

    thres_s_channel = np.zeros_like(s_channel)
    thres_s_channel[(s_channel >= COLOUR_THRES_MIN) & (s_channel <= COLOUR_THRES_MAX)] = 1

    thres_gradient = np.zeros_like(scaled_gradient_magnitude)
    thres_gradient[(scaled_gradient_magnitude >= MAGNITUDE_THRES_MIN) & (scaled_gradient_magnitude <= MAGNITUDE_THRES_MAX)] = 1

    thres_dir = np.zeros_like(gradient_dir)
    thres_dir[(gradient_dir >= DIR_THRES_MIN) & (gradient_dir <= DIR_THRES_MAX)] = 1

    # Combine the binary thresholds
    combined_binary = np.zeros_like(thres_sobel_x)
    combined_binary[((thres_sobel_x == 1) | (thres_s_channel == 1))] = 1

    if debug_mode:
        color_binary = np.dstack((np.zeros_like(thres_sobel_x), thres_sobel_x, thres_sobel_y, thres_s_channel)) * 255

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.set_title('Stacked thresholds')
        ax1.imshow(color_binary)

        ax2.set_title('Combined S channel and gradient thresholds')
        ax2.imshow(combined_binary, cmap='gray')
        cv2.waitKey(500)

    return combined_binary


def detect_and_fit_lines(image, left_lane, right_lane, debug_mode=False):

    # Find all nonzero pixels in image
    nonzero = image.nonzero()
    y_nonzero = np.array(nonzero[0])
    x_nonzero = np.array(nonzero[1])

    # Create a histogram of the bottom part of the image
    histogram = np.sum(image[image.shape[0] // 2:, :], axis=0)
    line_image = np.dstack((image, image, image)) * 255

    # Split the histogram into two halves down the midpoint
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = np.int(image.shape[0] // NUMBER_OF_WINDOWS)

    # Inital positions
    if left_lane.detected is False:
        left_lane.update_base(leftx_base)

    if right_lane.detected is False:
        right_lane.update_base(rightx_base)

    # Sliding window and Poly
    if left_lane.detected:
        left_lane.method = "Poly"
        leftx, lefty = search_around_poly(left_lane, x_nonzero, y_nonzero, debug_mode)
    else:
        left_lane.method = "Window"
        leftx, lefty, line_image = step_through_windows(line_image, window_height, left_lane, x_nonzero, y_nonzero,
                                                        debug_mode)

    if right_lane.detected:
        right_lane.method = "Poly"
        rightx, righty = search_around_poly(right_lane, x_nonzero, y_nonzero, debug_mode)

    else:
        right_lane.method = "Window"
        rightx, righty, line_image = step_through_windows(line_image, window_height, right_lane, x_nonzero, y_nonzero,
                                                          debug_mode)

    # Fit polynomial
    if len(leftx) and len(lefty):
        left_lane_fit = fit_polynomial(leftx, lefty)
        left_lane.update_fit(left_lane_fit)
        line_image[lefty, leftx] = [255, 0, 0]
    if len(rightx) and len(righty):
        right_lane_fit = fit_polynomial(rightx, righty)
        right_lane.update_fit(right_lane_fit)
        line_image[righty, rightx] = [0, 0, 255]

    left_fit = left_lane.current_fit
    right_fit = right_lane.current_fit
    left_base = left_fit[0]*image.shape[0]**2 + left_fit[1]*image.shape[0] + left_fit[2]
    right_base = right_fit[0]*image.shape[0]**2 + right_fit[1]*image.shape[0] + right_fit[2]
    left_lane.update_base(left_base)
    right_lane.update_base(right_base)

    if debug_mode:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # Generate x and y values for plotting
        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
        left_fitx = left_lane.current_fit[0] * ploty ** 2 + left_lane.current_fit[1] * ploty + left_lane.current_fit[2]
        right_fitx = right_lane.current_fit[0] * ploty ** 2 + right_lane.current_fit[1] * ploty + right_lane.current_fit[2]

        # Colors in the left and right lane regions
        line_image[lefty, leftx] = [255, 0, 0]
        line_image[righty, rightx] = [0, 0, 255]

        ax1.set_title('Perspective Image')
        ax1.imshow(line_image)
        # Plots the left and right polynomials on the lane lines
        ax1.plot(left_fitx, ploty, color='yellow')
        ax1.plot(right_fitx, ploty, color='yellow')

        ax2.set_title('Histogram')
        ax2.plot(histogram)
        cv2.waitKey(500)

    return line_image


def fit_polynomial(x_inds, y_inds):
    # Fit a second order polynomial to each using `np.polyfit`
    poly_lane = np.polyfit(y_inds, x_inds, 2)

    return poly_lane


def step_through_windows(image, window_height, lane, x_nonzero, y_nonzero, debug_mode=False):
    # Create empty lists to receive left and right lane pixel indices
    lane_pixels = []
    x_current = lane.x_best

    for window in range(NUMBER_OF_WINDOWS):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = image.shape[0] - (window+1)*window_height
        win_y_high = image.shape[0] - window*window_height
        win_x_low = x_current - WINDOW_MARGIN
        win_x_high = x_current + WINDOW_MARGIN

        if debug_mode:
            cv2.rectangle(image, (win_x_low, win_y_low),
                          (win_x_high, win_y_high), (0, 255, 0), 2)

        # Find the nonzero pixels within the window
        good_pixels = ((y_nonzero >= win_y_low) & (y_nonzero < win_y_high) &
                            (x_nonzero >= win_x_low) & (x_nonzero < win_x_high)).nonzero()[0]

        # Append these indices to the lists
        lane_pixels.append(good_pixels)

        # Recenter next window if enough pixels found
        if len(good_pixels) > MIN_NUMBER_PIXELS:
            x_base = np.int(np.mean(x_nonzero[good_pixels]))
            x_current = x_base

    lane_pixels = np.concatenate(lane_pixels)

    x_inds = x_nonzero[lane_pixels]
    y_inds = y_nonzero[lane_pixels]

    return x_inds, y_inds, image


def search_around_poly(lane, x_nonzero, y_nonzero, debug_mode=False):
    lane_inds = ((x_nonzero > (lane.current_fit[0] * (y_nonzero ** 2) + lane.current_fit[1] * y_nonzero +
                               lane.current_fit[2] - WINDOW_MARGIN)) &
                 (x_nonzero < (lane.current_fit[0] * (y_nonzero ** 2) + lane.current_fit[1] * y_nonzero +
                               lane.current_fit[2] + WINDOW_MARGIN)))

    # Again, extract left and right line pixel positions
    x_inds = x_nonzero[lane_inds]
    y_inds = y_nonzero[lane_inds]

    if len(x_inds) < 50:
        lane.detected = False

    return x_inds, y_inds


def draw_lanes_on_image(left_lane, right_lane, transformed_image, perspective_matrix, undistorted_image):
    left_fit = left_lane.current_fit
    right_fit = right_lane.current_fit

    # Calculate lane curvature
    left_lane.radius_of_covergence()
    right_lane.radius_of_covergence()

    # Calculate lane center
    width = transformed_image.shape[1]
    left_lane.update_center(width)
    right_lane.update_center(width)

    # Plotting parameters
    ploty = np.linspace(0, transformed_image.shape[0] - 1, transformed_image.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(transformed_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, perspective_matrix, (transformed_image.shape[1],
                                                                   transformed_image.shape[0]))
    # Combine the result with the original image
    left_poly = (round(left_lane.current_fit[0], 2), round(left_lane.current_fit[1], 2), round(left_lane.current_fit[2], 2))
    right_poly = (round(right_lane.current_fit[0], 2), round(right_lane.current_fit[1], 2), round(right_lane.current_fit[2], 2))

    result = cv2.addWeighted(undistorted_image, 1, newwarp, 0.3, 0)
    result = cv2.putText(result, f"Left: {left_lane.method} {left_poly} Right: {right_lane.method} {right_poly}",
                         org=(50, 60), fontFace=2, fontScale=1, color=(255, 255, 255), thickness=2)
    # result = cv2.putText(result, f"Left Radius: {round(left_lane.radius_of_curvature, 1)} "
    #                              f"Right Radius: {round(right_lane.radius_of_curvature, 1)}",
    #                      org=(50, 120), fontFace=2, fontScale=1, color=(255, 255, 255), thickness=2)
    # result = cv2.putText(result, f"Left Center: {round(left_lane.center, 1)} "
    #                              f"Right Center: {round(right_lane.center, 1)}",
    #                      org=(50, 180), fontFace=2, fontScale=1, color=(255, 255, 255), thickness=2)
    return result


def find_lanes(image, mtx, dist, left_lane, right_lane, pers_M, invpers_M, debug_mode=False):
    # Undistort using camera calibration
    undistorted_image = cv2.undistort(image, mtx, dist, None, mtx)

    # Colour and Gradient Thresholding
    thresholded_image = threshold_image(undistorted_image, debug_mode)

    # Perspective transform
    transformed_image = perspective_transform(thresholded_image, pers_M, debug_mode)

    # Find lines
    lanes_image = detect_and_fit_lines(transformed_image, left_lane, right_lane, debug_mode)

    # Draw lanes on image
    res_image = draw_lanes_on_image(left_lane, right_lane, transformed_image, invpers_M, undistorted_image)

    return res_image

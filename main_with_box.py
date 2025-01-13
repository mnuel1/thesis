import numpy as np
import cv2
import pysift
import slow
from matplotlib import pyplot as plt
import logging
import time  # Import time module

logger = logging.getLogger(__name__)

MIN_MATCH_COUNT = 10
sift = cv2.SIFT_create()

# Start the timer
start_time = time.time()

img1 = cv2.imread('images/1.jpg', 0)   # queryImage
img2 = cv2.imread('images/2.jpg', 0)  # trainImage

# Check if images were loaded correctly
if img1 is None or img2 is None:
    print("Error loading images. Please check file paths.")
    exit(1)

# Compute SIFT keypoints and descriptors
# kp1, des1 = slow.computeKeypointsAndDescriptors(img1)
# kp2, des2 = slow.computeKeypointsAndDescriptors(img2)

# # Compute SIFT keypoints and descriptors
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Initialize and use FLANN    
index_params = dict(algorithm=0, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Lowe's ratio test
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    # Estimate homography between template and scene
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]

    # Define bounding box in the template image
    h, w = img1.shape
    pts = np.float32([[0, 0],
                    [0, h - 1],
                    [w - 1, h - 1],
                    [w - 1, 0]]).reshape(-1, 1, 2)

    # Transform points to the scene image
    dst = cv2.perspectiveTransform(pts, M)

    # Compute YOLO bounding box format
    x_min = np.min(dst[:, 0, 0])
    x_max = np.max(dst[:, 0, 0])
    y_min = np.min(dst[:, 0, 1])
    y_max = np.max(dst[:, 0, 1])

    # Width and height of the scene image
    img_width, img_height = img2.shape[1], img2.shape[0]

    # Normalize coordinates and compute center and dimensions
    x_center = ((x_min + x_max) / 2) / img_width
    y_center = ((y_min + y_max) / 2) / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    print(x_center, y_center, width, height)
else:
    print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))

# End the timer and calculate elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.2f} seconds")

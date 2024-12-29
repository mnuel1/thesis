import os
import zipfile
import uuid
import numpy as np
import cv2
import pysift
import logging
from pyunpack import Archive

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# Set up logging
logger = logging.getLogger(__name__)

# Constants
MIN_MATCH_COUNT = 10
FLANN_INDEX_KDTREE = 0

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Helper function to process images and compute bounding boxes
def process_images(base_image_path, image_paths):
    results = []
    img1 = cv2.imread(base_image_path, 0)  # Base query image

    # Compute SIFT keypoints and descriptors for the base image
    kp1, des1 = pysift.computeKeypointsAndDescriptors(img1)

    for image_path in image_paths:
        img2 = cv2.imread(image_path, 0)  # Current image to compare
        if img2 is None:
            logger.error(f"Could not read image: {image_path}")
            continue

        # Compute SIFT keypoints and descriptors for the current image
        kp2, des2 = pysift.computeKeypointsAndDescriptors(img2)

        # Initialize FLANN matcher
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Apply Lowe's ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if len(good) > MIN_MATCH_COUNT:
            # Estimate homography between base and current image
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]

            # Define bounding box in the base image
            h, w = img1.shape
            pts = np.float32([[0, 0],
                              [0, h - 1],
                              [w - 1, h - 1],
                              [w - 1, 0]]).reshape(-1, 1, 2)

            # Transform points to the current image
            dst = cv2.perspectiveTransform(pts, M)

            # Compute YOLO bounding box format
            x_min = np.min(dst[:, 0, 0])
            x_max = np.max(dst[:, 0, 0])
            y_min = np.min(dst[:, 0, 1])
            y_max = np.max(dst[:, 0, 1])

            # Dimensions of the current image
            img_width, img_height = img2.shape[1], img2.shape[0]

            # Normalize coordinates and compute center and dimensions
            x_center = ((x_min + x_max) / 2) / img_width
            y_center = ((y_min + y_max) / 2) / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height

            # Class index (assuming class ID 0 for this example)
            class_id = 0

            # Append result
            results.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        else:
            logger.warning(f"Not enough matches found for image: {image_path}")

    return results

# API Endpoint to process images
@app.route('/run-sift', methods=['POST'])
def generate_bounding_box():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Ensure uploaded file is a zip file
    if not (file.filename.endswith('.zip') or file.filename.endswith('.rar')):
        return jsonify({"error": "Only ZIP or RAR files are allowed"}), 400
    
    # Create a temporary directory to extract the zip file
    random_uid = str(uuid.uuid4())
    upload_folder = f"temp_upload_{random_uid}"
    os.makedirs(upload_folder, exist_ok=True)
    archive_path = os.path.join(upload_folder, file.filename)
    
    # Save the uploaded file
    file.save(archive_path)

    # Extract the archive
    try:
        Archive(archive_path).extractall(upload_folder)
    except Exception as e:
        return jsonify({"error": f"Failed to extract archive: {str(e)}"}), 400

    # # Find the base image and other images
    # base_image_path = None
    # image_paths = []
    # for root, dirs, files in os.walk(upload_folder):
    #     for filename in files:
    #         if filename == "base_image":
    #             base_image_path = os.path.join(root, filename)
    #         else:
    #             image_paths.append(os.path.join(root, filename))

    # if not base_image_path:
    #     return jsonify({"error": "Base image (base_image) not found in the zip"}), 400

    # # Process images and generate results
    # results = process_images(base_image_path, image_paths)

    # # Save results to a text file
    # output_file = os.path.join(upload_folder, "bounding_boxes.txt")
    # with open(output_file, 'w') as f:
    #     f.write("\n".join(results))

    # # Return the text file as response
    # return send_file(output_file, as_attachment=True, download_name="bounding_boxes.txt")
    return jsonify("success"), 200

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)

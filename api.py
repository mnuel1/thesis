import os
import zipfile
import json
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
TEMP_UPLOAD_DIR = "temp_upload"

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Helper function to process images and compute bounding boxes
def process_images(base_image_path, target_image_path):
    img1 = cv2.imread(base_image_path, 0)  # base image 
    img2 = cv2.imread(target_image_path, 0)  # target image
    # Compute SIFT keypoints and descriptors
    kp1, des1 = pysift.computeKeypointsAndDescriptors(img1)
    kp2, des2 = pysift.computeKeypointsAndDescriptors(img2)

    # Initialize and use FLANN
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
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

        return x_center, y_center, width, height
    else:
        return None  # Not enough matches found


def validate_request(base_file, target_file):

    if base_file.filename == '' or target_file.filename == '':
        return 1, "No file selected"
        
    if not ((base_file.filename.endswith('.zip') or base_file.filename.endswith('.rar')) 
            or target_file.filename.endswith('.zip') or target_file.filename.endswith('.rar')):
        return 1, "Only ZIP or RAR files are allowed"

    return 0, "" # no error

def generate_id():
    return str(uuid.uuid4())

def create_directory():
    random_uid = generate_id()
    upload_folder = f"{TEMP_UPLOAD_DIR}/{random_uid}"
    upload_folder_base = f"{upload_folder}/base_images"
    upload_folder_target = f"{upload_folder}/target_images"
    os.makedirs(upload_folder_base, exist_ok=True)
    os.makedirs(upload_folder_target, exist_ok=True)

    return upload_folder, upload_folder_base, upload_folder_target

def load_json(class_data_str):
    try:
        # Parse the JSON string into a Python dictionary
        class_data = json.loads(class_data_str)
    except json.JSONDecodeError:
        return 1, "Invalid JSON format for class."

    # Access the 'class_values' dictionary from the parsed data
    class_values = class_data.get('class_values')

    if class_values is None:
        return 1, "class_values key is missing in the provided data."
    
    return 0, class_values

@app.route('/run-sift', methods=['POST'])
def generate_bounding_box():

    if 'base_archive' not in request.files or 'target_archive' not in request.files:
        return jsonify({"error": "Both 'base_archive' and 'target_archive' are required."}), 400
    
    if 'class' not in request.form:
        return jsonify({"error": "Object Class is required."}), 400
    
    class_data_str = request.form['class']

    json_error, class_values = load_json(class_data_str)
    if json_error:
        return jsonify({"error": class_values}), 400
               
    base_archive = request.files['base_archive']
    target_archive = request.files['target_archive']
    
    upload_error, message = validate_request(base_archive, target_archive)
    if upload_error:
        return jsonify({"error": message}), 400
           
    # Create a temporary directory to extract the zip file
    upload_folder, upload_folder_base, upload_folder_target = create_directory()

    base_path = os.path.join(upload_folder_base, base_archive.filename)
    target_path = os.path.join(upload_folder_target, target_archive.filename)
    
    # Save the uploaded file
    base_archive.save(base_path)
    target_archive.save(target_path)
    
    # Extract the archive
    try:
        Archive(base_path).extractall(upload_folder_base)
        Archive(target_path).extractall(upload_folder_target)
    except Exception as e:
        return jsonify({"error": f"Failed to extract archive: {str(e)}"}), 400

    # Process base images
    bounding_boxes = []
    for base_filename, class_id in class_values.items():
        base_image_path = os.path.join(upload_folder_base, base_filename)
        print("here")
        # Run SIFT on all target images for the current base image
        for target_filename in os.listdir(upload_folder_target):
            target_image_path = os.path.join(upload_folder_target, target_filename)
            print(target_image_path)
            # Generate bounding box for this base-target image pair
            if os.path.exists(base_image_path) and os.path.exists(target_image_path):
                bbox = process_images(base_image_path, target_image_path)
                print(bbox)
                if bbox:
                    x_center, y_center, width, height = bbox
                    bounding_boxes.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                else:
                    bounding_boxes.append(f"{class_id} 0 0 0 0")  # No match found, set to 0s

    # Save results to text file
    output_file = os.path.join(upload_folder, "bounding_boxes.txt")
    with open(output_file, 'w') as f:
        f.write("\n".join(bounding_boxes))

    # Return the bounding boxes text file
    # return send_file(output_file, as_attachment=True, download_name="bounding_boxes.txt")
    return jsonify(
        { 
            "message": "success" 
        }
    ), 200

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)

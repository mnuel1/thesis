import os
import zipfile
import json
import uuid
import numpy as np
import cv2
import pysift
import logging
import shutil
from pyunpack import Archive

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# Set up logging
logger = logging.getLogger(__name__)

# Constants
MIN_MATCH_COUNT = 10
FLANN_INDEX_KDTREE = 0
TEMP_UPLOAD_DIR = "temp_upload"
DOMAIN_LINK = "http://127.0.0.1:5000/"

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Helper function to process images and compute bounding boxes
def process_images(base_image_path, target_image_path, type):

    # return 0.499512, 0.499335, 0.999023, 0.998670
    sift = cv2.SIFT_create()
    img1 = cv2.imread(base_image_path, 0)  # base image 
    img2 = cv2.imread(target_image_path, 0)  # target image

    # Compute SIFT keypoints and descriptors
    kp1, des1 = pysift.computeKeypointsAndDescriptors(img1) if type == 1 else sift.detectAndCompute(img1, None)
    kp2, des2 = pysift.computeKeypointsAndDescriptors(img2) if type == 1 else sift.detectAndCompute(img2, None)

    # Initialize and use FLANN    
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
    annotated_folder = f"{upload_folder}/annotated"
    os.makedirs(upload_folder_base, exist_ok=True)
    os.makedirs(upload_folder_target, exist_ok=True)
    os.makedirs(annotated_folder, exist_ok=True)

    return random_uid, upload_folder, upload_folder_base, upload_folder_target, annotated_folder

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

    if 'base_archive' not in request.files or 'target_archive' not in request.files or 'label' not in request.files:
        return jsonify({"error": "Both 'base_archive' and 'target_archive' and 'label' are required."}), 400
    
    if 'class' not in request.form:
        return jsonify({"error": "Object Class is required."}), 400
    
    class_data_str = request.form['class']
    type = request.form['version']

    json_error, class_values = load_json(class_data_str)
    if json_error:
        return jsonify({"error": class_values}), 400
               
    base_archive = request.files['base_archive']
    target_archive = request.files['target_archive']
    label_txt = request.files['label']
    
    upload_error, message = validate_request(base_archive, target_archive)
    if upload_error:
        return jsonify({"error": message}), 400
           
    # Create a temporary directory to extract the zip file
    random_uid, upload_folder, upload_folder_base, upload_folder_target, annotated_folder = create_directory()

    base_path = os.path.join(upload_folder_base, base_archive.filename)
    target_path = os.path.join(upload_folder_target, target_archive.filename)
    label_path = os.path.join(annotated_folder, "classes.txt")
    
    # Save the uploaded file
    base_archive.save(base_path)
    target_archive.save(target_path)
    label_txt.save(label_path)

    # Extract the archive
    try:
        Archive(base_path).extractall(upload_folder_base)
        Archive(target_path).extractall(upload_folder_target)

        # save storage by deleting
        os.remove(base_path)
        os.remove(target_path)
    except Exception as e:
        return jsonify({"error": f"Failed to extract archive: {str(e)}"}), 400

    # Process base images
    success = 0
    fail = 0

    # Get the total number of target images
    total_images = len(os.listdir(upload_folder_target))

    for target_filename in os.listdir(upload_folder_target):
        target_image_path = os.path.join(upload_folder_target, target_filename)
        
        # Check if the target image exists
        if os.path.exists(target_image_path):
            bounding_boxes = []  # Reset bounding_boxes for each target image
            
            # Compare the current target image with all base images
            for base_filename, class_id in class_values.items():
                base_image_path = os.path.join(upload_folder_base, base_filename)
                
                # Check if the base image exists
                if os.path.exists(base_image_path):
                    # Process the images and generate bounding box
                    bbox = process_images(base_image_path, target_image_path, type)
                                        
                    # If a bounding box was found, append it to the list
                    if bbox:
                        x_center, y_center, width, height = bbox
                        bounding_boxes.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                        success += 1
                    else:
                        # No match found, append a 0s bounding box
                        bounding_boxes.append(f"{class_id} 0 0 0 0")
                        fail += 1
            
            # Now that all comparisons for the current target image are done, write the results to a .txt file
            target_name_without_ext = os.path.splitext(target_filename)[0]
            output_file = os.path.join(annotated_folder, f"{target_name_without_ext}.txt")
            
            # Write all bounding boxes for the current target image
            with open(output_file, 'w') as f:
                f.write("\n".join(bounding_boxes))

    # Calculate accuracy
    accuracy = (success / (success + fail)) * 100 if (success + fail) > 0 else 0

    # archive the output
    archive_path = os.path.join(upload_folder, f"{random_uid}.zip")
    shutil.make_archive(archive_path.replace('.zip', ''), 'zip', upload_folder)

    # Return the results as a JSON response
    return jsonify(
        {
            "message": "success",
            "total_images": total_images,
            "total_success": success,
            "total_fail": fail,
            "accuracy": f"{accuracy:.2f}%",
            "download_url": f"{DOMAIN_LINK}download_archive/{random_uid}"
        }
    ), 200


@app.route('/download_archive/<random_uid>', methods=['GET'])
def download_archive(random_uid):    
    
    archive_path = os.path.join(TEMP_UPLOAD_DIR, random_uid, f"{random_uid}.zip")
    
    # Check if the file exists before attempting to serve it
    if not os.path.exists(archive_path):
        return jsonify({"message": "File not found"}), 404
        
    return send_file(archive_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)

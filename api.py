import os
import zipfile
import json
import uuid
import numpy as np
import cv2
import logging
import shutil
import concurrent.futures
from pyunpack import Archive
from functools import lru_cache
from collections import defaultdict
import time
import statistics  # Import statistics module for calculating mean

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from threading import Lock
import threading

from asgiref.wsgi import WsgiToAsgi

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MIN_MATCH_COUNT = 8  # Reduced from 10 to improve detection rate
FLANN_INDEX_KDTREE = 1  # Increased for better accuracy
TEMP_UPLOAD_DIR = "temp_upload"
DOMAIN_LINK = "http://127.0.0.1:5000/"
MAX_WORKERS = 4  # Number of parallel workers for processing
CONFIDENCE_THRESHOLD = 0.65  # Threshold for confidence in matches
SIFT_FEATURES = 3000  # Maximum number of features to detect
RATIO_TEST_THRESHOLD = 0.75  # Threshold for Lowe's ratio test

# Initialize Flask app
app = Flask(__name__)
CORS(app)

asgi_app = WsgiToAsgi(app)

progress_data = {}
progress_lock = Lock()


# Cache for SIFT descriptors to avoid recomputation
@lru_cache(maxsize=100)
def compute_sift_features(image_path):
    """Compute and cache SIFT features for an image"""
    try:
        img = cv2.imread(image_path, 0)
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return None, None
        
        # img = cv2.equalizeHist(img)
        
        img = cv2.GaussianBlur(img, (3, 3), 0)
            
        # Create SIFT detector with custom parameters for better performance
        sift = cv2.SIFT_create(
            nfeatures=SIFT_FEATURES,
            nOctaveLayers=4,
            contrastThreshold=0.03,
            edgeThreshold=12,
            sigma=1.6
        )
        
        # Detect keypoints and compute descriptors
        kp, des = sift.detectAndCompute(img, None)
        return img, kp, des
    except Exception as e:
        logger.error(f"Error computing SIFT features for {image_path}: {str(e)}")
        return None, None, None

def process_images(base_image_path, target_image_path, type):
    """Process images and compute bounding boxes with improved matching"""
    try:
        # Load images and compute features
        img1, kp1, des1 = compute_sift_features(base_image_path)
        img2, kp2, des2 = compute_sift_features(target_image_path)
        
        if img1 is None or img2 is None or des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
            return None
            
        # Initialize FLANN matcher with optimized parameters
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # Increased checks for better accuracy
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Use knnMatch for better matching
        matches = flann.knnMatch(des1, des2, k=2)
        
        if not matches or len(matches) < MIN_MATCH_COUNT:
            return None
            
        # Apply Lowe's ratio test with optimized threshold
        good = []
        for m, n in matches:
            if m.distance < RATIO_TEST_THRESHOLD * n.distance:
                good.append(m)
                
        # Enhanced match filtering - require more good matches for high confidence
        if len(good) > MIN_MATCH_COUNT:
            # Estimate homography between template and scene
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            
            # Use RANSAC with optimized threshold
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
            if M is None:
                return None
                
            # Count inliers (matched points that satisfy the homography)
            matchesMask = mask.ravel().tolist()
            inliers = sum(matchesMask)
            
            # Calculate confidence based on inliers ratio
            confidence = inliers / len(good) if len(good) > 0 else 0
            
            # Only proceed if confidence is high enough
            if confidence < CONFIDENCE_THRESHOLD:
                return None
                
            # Define bounding box in the template image
            h, w = img1.shape
            pts = np.float32([[0, 0],
                            [0, h - 1],
                            [w - 1, h - 1],
                            [w - 1, 0]]).reshape(-1, 1, 2)
                            
            # Transform points to the scene image
            dst = cv2.perspectiveTransform(pts, M)
            
            # Compute YOLO bounding box format with better clipping
            x_min = max(0, np.min(dst[:, 0, 0]))
            x_max = min(img2.shape[1], np.max(dst[:, 0, 0]))
            y_min = max(0, np.min(dst[:, 0, 1]))
            y_max = min(img2.shape[0], np.max(dst[:, 0, 1]))
            
            # Check if the bounding box is valid (not too small or too large)
            if (x_max - x_min) < 5 or (y_max - y_min) < 5:
                return None
                
            if (x_max - x_min) > img2.shape[1] * 0.9 and (y_max - y_min) > img2.shape[0] * 0.9:
                return None
                
            # Width and height of the scene image
            img_width, img_height = img2.shape[1], img2.shape[0]
            
            # Normalize coordinates and compute center and dimensions
            x_center = ((x_min + x_max) / 2) / img_width
            y_center = ((y_min + y_max) / 2) / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height
            
            # Return normalized bounding box
            return x_center, y_center, width, height, confidence
    except Exception as e:
        logger.error(f"Error in process_images: {str(e)}")
        return None
        
    return None  # Not enough matches found

def validate_request(base_file, target_file):
    """Validate uploaded files"""
    if base_file.filename == '' or target_file.filename == '':
        return 1, "No file selected"
        
    allowed_extensions = ('.zip', '.rar')
    if not (base_file.filename.lower().endswith(allowed_extensions) and 
            target_file.filename.lower().endswith(allowed_extensions)):
        return 1, "Only ZIP or RAR files are allowed"

    return 0, ""  # no error

def generate_id():
    """Generate a unique ID for the upload directory"""
    return str(uuid.uuid4())

def create_directory():
    """Create necessary directories for processing"""
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
    """Load and validate class JSON data"""
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

def process_target_image(args, random_uid):
    """Process a single target image against all base images"""
    target_image_path, base_images, class_values, type = args
    bounding_boxes = []
    success_count = 0
    confidences = []  # Track all confidence scores
    
    target_name = os.path.basename(target_image_path)
    logger.info(f"Processing target image: {target_name}")
    
    # We'll track the best match for each class
    best_matches = {}
    
    # First, load the target image once to avoid loading it multiple times
    target_img, target_kp, target_des = compute_sift_features(target_image_path)
    if target_img is None:
        logger.warning(f"Could not load target image: {target_name}")
        return target_image_path, bounding_boxes, 0, []  # Return empty confidence list
    
    # Process against each base image
    for base_filename, class_id in class_values.items():
        base_image_path = os.path.join(base_images, base_filename)
        
        # Check if the base image exists
        if os.path.exists(base_image_path):
            # Process the images and generate bounding box
            try:
                # Get base image features
                base_img, base_kp, base_des = compute_sift_features(base_image_path)
                
                if base_img is None or base_des is None or target_des is None:
                    continue
                    
                # Initialize FLANN matcher with optimized parameters
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                
                # Use knnMatch for better matching
                matches = flann.knnMatch(base_des, target_des, k=2)
                
                if not matches or len(matches) < MIN_MATCH_COUNT:
                    continue
                    
                # Apply Lowe's ratio test with optimized threshold
                good = []
                for m, n in matches:
                    if m.distance < RATIO_TEST_THRESHOLD * n.distance:
                        good.append(m)
                        
                # Enhanced match filtering - require more good matches for high confidence
                if len(good) > MIN_MATCH_COUNT:
                    # Estimate homography between template and scene
                    src_pts = np.float32([base_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([target_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                    
                    # Use RANSAC with optimized threshold
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
                    if M is None:
                        continue
                        
                    # Count inliers (matched points that satisfy the homography)
                    matchesMask = mask.ravel().tolist()
                    inliers = sum(matchesMask)
                    
                    # Calculate confidence based on inliers ratio
                    confidence = inliers / len(good) if len(good) > 0 else 0
                    
                    # Only proceed if confidence is high enough
                    if confidence < CONFIDENCE_THRESHOLD:
                        continue
                        
                    # Define bounding box in the template image
                    h, w = base_img.shape
                    pts = np.float32([[0, 0],
                                    [0, h - 1],
                                    [w - 1, h - 1],
                                    [w - 1, 0]]).reshape(-1, 1, 2)
                                    
                    # Transform points to the scene image
                    dst = cv2.perspectiveTransform(pts, M)
                    
                    # Compute YOLO bounding box format with better clipping
                    x_min = max(0, np.min(dst[:, 0, 0]))
                    x_max = min(target_img.shape[1], np.max(dst[:, 0, 0]))
                    y_min = max(0, np.min(dst[:, 0, 1]))
                    y_max = min(target_img.shape[0], np.max(dst[:, 0, 1]))
                    
                    # Check if the bounding box is valid (not too small or too large)
                    if (x_max - x_min) < 5 or (y_max - y_min) < 5:
                        continue
                        
                    # Reject if object covers too much of the image (likely false positive)
                    if (x_max - x_min) > target_img.shape[1] * 0.9 and (y_max - y_min) > target_img.shape[0] * 0.9:
                        continue
                        
                    # Width and height of the scene image
                    img_width, img_height = target_img.shape[1], target_img.shape[0]
                    
                    # Normalize coordinates and compute center and dimensions
                    x_center = ((x_min + x_max) / 2) / img_width
                    y_center = ((y_min + y_max) / 2) / img_height
                    width = (x_max - x_min) / img_width
                    height = (y_max - y_min) / img_height
                    
                    # Track this match
                    match_info = {
                        'bbox': (x_center, y_center, width, height),
                        'confidence': confidence,
                        'inliers': inliers,
                        'class_id': class_id
                    }
                    
                    # Store best match for this class
                    if class_id not in best_matches or confidence > best_matches[class_id]['confidence']:
                        best_matches[class_id] = match_info
                    
            except Exception as e:
                logger.error(f"Error processing base image {base_filename}: {str(e)}")
    
    curr_confidence = 0
    # Add the best match for each class to our results
    for class_id, match in best_matches.items():
        x_center, y_center, width, height = match['bbox']
        confidence = match['confidence']
        bounding_boxes.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        confidences.append(confidence)  # Store the confidence score
        success_count += 1
        logger.info(f"Found object class {class_id} in {target_name} with confidence {confidence:.4f}")
        curr_confidence = confidence
    
    with progress_lock:
        progress_data[random_uid]["processed"] += 1
        progress_data[random_uid]["details"].append({
            "image": target_name,
            "confidence": curr_confidence,
            "status": "done"
        })
          
    return target_image_path, bounding_boxes, success_count, confidences

def process_images_background(process_args, class_values, annotated_folder, total_images, upload_folder, random_uid, start_time):
    # Process images in parallel
    success = 0
    fail = 0
    all_confidences = []  # To collect all confidence scores
    
    # Track per-image success for better accuracy calculation
    images_with_detections = 0
        
    # Process images in parallel using ThreadPoolExecutor
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_path = {executor.submit(process_target_image, arg, random_uid): arg for arg in process_args}
        
        for future in concurrent.futures.as_completed(future_to_path):
            try:
                target_path, bounding_boxes, successes, confidences = future.result()
                results[target_path] = bounding_boxes
                success += successes
                all_confidences.extend(confidences)  # Collect all confidence scores
                
                # If any detections were made for this image, count it as a success
                if successes > 0:
                    images_with_detections += 1
                # Calculate failures based on number of classes and successes
                fail += len(class_values) - successes
            except Exception as e:
                logger.error(f"Error processing task: {str(e)}")
                fail += len(class_values)  # Count all as failures if exception occurs
    
    # Write results to files
    for target_path, bounding_boxes in results.items():
        target_name_without_ext = os.path.splitext(os.path.basename(target_path))[0]
        output_file = os.path.join(annotated_folder, f"{target_name_without_ext}.txt")
        
        # Write all bounding boxes for the current target image
        with open(output_file, 'w') as f:
            f.write("\n".join(bounding_boxes))

    # Calculate standard accuracy (per detection)
    total_attempts = success + fail
    
    # Calculate new detection_accuracy based on average confidence
    detection_accuracy = 0
    if all_confidences:
        # Calculate the average confidence score and convert to percentage
        avg_confidence = statistics.mean(all_confidences) * 100
        detection_accuracy = avg_confidence
    
    # Calculate image-based accuracy (percentage of images with at least one detection)
    image_accuracy = (images_with_detections / total_images) * 100 if total_images > 0 else 0

    # Archive the output
    archive_path = os.path.join(upload_folder, f"{random_uid}.zip")
    shutil.make_archive(archive_path.replace('.zip', ''), 'zip', upload_folder)

    # Calculate processing time
    processing_time = time.time() - start_time

    result_path = os.path.join(upload_folder, f"{random_uid}.json")
    with open(result_path, 'w') as f:
        json.dump({
            "message": "success",
            "total_images": total_images,
            "images_with_detections": images_with_detections,
            "detection_accuracy": f"{detection_accuracy:.2f}%",
            "total_annotated_images": f"{image_accuracy:.2f}%",
            "processing_time": f"{processing_time:.2f} seconds",
            "download_url": f"{DOMAIN_LINK}download_archive/{random_uid}"
        }, f)
   
@app.route('/run-sift', methods=['POST'])
def generate_bounding_box():    
    """Main route for processing images and generating bounding boxes"""
    start_time = time.time()
    
    if 'base_archive' not in request.files or 'target_archive' not in request.files or 'label' not in request.files:
        return jsonify({"error": "Both 'base_archive' and 'target_archive' and 'label' are required."}), 400
    
    if 'class' not in request.form:
        return jsonify({"error": "Object Class is required."}), 400
    
    class_data_str = request.form['class']
    type = int(request.form.get('version', 0))  # Default to 0 if not provided

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
        logger.error(f"Archive extraction error: {str(e)}")
        return jsonify({"error": f"Failed to extract archive: {str(e)}"}), 400

    # Get list of target images
    target_images = [os.path.join(upload_folder_target, f) for f in os.listdir(upload_folder_target) 
                     if os.path.isfile(os.path.join(upload_folder_target, f)) and 
                     f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    # Count total images
    total_images = len(target_images)
    
    if total_images == 0:
        return jsonify({"error": "No valid images found in target archive"}), 400

    with progress_lock:
        progress_data[random_uid] = {
            "total": total_images,
            "processed":  0,
            "details": []
            
        }
    # Prepare arguments for parallel processing
    process_args = [(img_path, upload_folder_base, class_values, type) for img_path in target_images]
    
    threading.Thread(target=process_images_background, args=(
        process_args, class_values, annotated_folder, total_images, upload_folder, random_uid, start_time
    )).start()

    return jsonify({
        "message": "processing started",
        "uid": random_uid,
        "status_url": f"{DOMAIN_LINK}get-results/{random_uid}",
        "progress_url": f"{DOMAIN_LINK}progress/{random_uid}"
    }), 202
    

@app.route('/download_archive/<random_uid>', methods=['GET'])
def download_archive(random_uid):    
    """Route for downloading the results archive"""
    archive_path = os.path.join(TEMP_UPLOAD_DIR, random_uid, f"{random_uid}.zip")
    
    # Check if the file exists before attempting to serve it
    if not os.path.exists(archive_path):
        return jsonify({"message": "File not found"}), 404
        
    return send_file(archive_path, as_attachment=True)

@app.route('/get-results/<random_uid>', methods=['GET'])
def get_results(random_uid):
    result_path = os.path.join(TEMP_UPLOAD_DIR, f"{random_uid}\{random_uid}.json")
    
    if not os.path.exists(result_path):
        return jsonify({"status": "processing"}), 202

    with open(result_path, 'r') as f:
        result = json.load(f)
    
    if "error" in result:
        return jsonify(result), 500
    
    return jsonify(result), 200

@app.route('/progress/<random_uid>', methods=['GET'])
def get_progress(random_uid):
    with progress_lock:
        data = progress_data.get(random_uid)
        if data is None:
            return jsonify({"error": "UID not found or not started"}), 404
        return jsonify(data), 200


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok", "version": "1.0.0"}), 200


# Cleanup function to remove old temporary files
def cleanup_temp_directories(max_age_hours=24):
    """Clean up temporary directories older than max_age_hours"""
    try:
        if not os.path.exists(TEMP_UPLOAD_DIR):
            return
            
        current_time = time.time()
        for dir_name in os.listdir(TEMP_UPLOAD_DIR):
            dir_path = os.path.join(TEMP_UPLOAD_DIR, dir_name)
            if os.path.isdir(dir_path):
                creation_time = os.path.getctime(dir_path)
                if (current_time - creation_time) // 3600 > max_age_hours:
                    shutil.rmtree(dir_path, ignore_errors=True)
    except Exception as e:
        logger.error(f"Error cleaning up temp directories: {str(e)}")


if __name__ == "__main__":
    # Create temp directory if it doesn't exist
    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
    
    # Clean up old files on startup
    cleanup_temp_directories()
    
    # Run the Flask app
    import uvicorn
    uvicorn.run(asgi_app, host="0.0.0.0", port=8000)
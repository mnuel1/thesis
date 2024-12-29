import cv2
import numpy as np
import os

def annotate_image(base_image_path, search_images_folder, output_folder):
    
    base_img = cv2.imread(base_image_path, cv2.IMREAD_GRAYSCALE)
    if base_img is None:
        raise ValueError("Error loading base image.")
    
    sift = cv2.SIFT_create()
    
    base_keypoints, base_descriptors = sift.detectAndCompute(base_img, None)
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    for image_file in os.listdir(search_images_folder):
        if image_file.endswith(('.jpg', '.png', '.jpeg')):  
            
            search_img = cv2.imread(os.path.join(search_images_folder, image_file), cv2.IMREAD_GRAYSCALE)
            if search_img is None:
                print(f"Error loading image: {image_file}")
                continue
            
            search_keypoints, search_descriptors = sift.detectAndCompute(search_img, None)
            
            matches = flann.knnMatch(base_descriptors, search_descriptors, k=2)
            
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:  
                    good_matches.append(m)

            
            if len(good_matches) > 10:  
                src_pts = np.float32([base_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([search_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                
                h, w = base_img.shape
                base_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

                
                if M is not None:
                    dst_corners = cv2.perspectiveTransform(base_corners, M)

                    
                    search_img_color = cv2.cvtColor(search_img, cv2.COLOR_GRAY2BGR)
                    cv2.polylines(search_img_color, [np.int32(dst_corners)], isClosed=True, color=(0, 255, 0), thickness=2)

                    
                    output_path = os.path.join(output_folder, f"annotated_{image_file}")
                    cv2.imwrite(output_path, search_img_color)
                    print(f"Annotated image saved: {output_path}")


base_image_path = 'box.png'  
search_images_folder = './input_dataset'  
output_folder = './output_dataset'  

os.makedirs(output_folder, exist_ok=True)

annotate_image(base_image_path, search_images_folder, output_folder)

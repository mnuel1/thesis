from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load a pretrained YOLOv5 or YOLOv8 model
model = YOLO('yolov5s.pt')  # or 'yolov8n.pt' for YOLOv8

# Run inference on an image
results = model('test/food/train/apple_r.png')  # replace with your image path

# Show results
results[0].show()  # opens image with bounding boxes

# Save results
results[0].save(filename='output.jpg')  # saves image with boxes

# Print detections
for result in results:
    boxes = result.boxes
    for box in boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        label = model.names[cls_id]
        print(f"{label}: {conf:.2f}")

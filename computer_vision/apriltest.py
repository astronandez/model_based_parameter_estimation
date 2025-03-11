import cv2
from apriltag import apriltag
import numpy as np

# Test image (change path accordingly)
image_path = "test_apriltags.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
detector = apriltag("tagStandard41h12")

# Detect AprilTags
detections = detector.detect(image)

print(f"Detected {len(detections)} tags.")
for det in detections:
    print(f"Tag ID: {det.tag_id}, Center: {det.center}")
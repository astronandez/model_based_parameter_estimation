import cv2
import platform
from pupil_apriltags import Detector
from numpy import array, float32, mean, round
from tools.common import loadConfig
import matplotlib.pyplot as plt

system_name = platform.system()
detector1 = Detector(families="tagStandard41h12",
                     nthreads=1,
                     quad_decimate=1.0,
                     quad_sigma=0.0,
                     refine_edges=1,
                     decode_sharpening=0.25,
                     debug=0) 

detector2 = Detector(families="tagCircle49h12",
                     nthreads=1,
                     quad_decimate=1.0,
                     quad_sigma=0.0,
                     refine_edges=1,
                     decode_sharpening=0.25,
                     debug=0) 


calibration_data = loadConfig("./configuration_files/camera_configs/calibration_matrix_dslr.json")
camera_matrix = array(calibration_data["camera_matrix"])
dist_coeffs = array(calibration_data["dist_coeff"])

scale_x = 640 / 1920
scale_y = 360 / 1080
f_x = camera_matrix[0, 0] * scale_x  # Focal length in X
c_x = camera_matrix[0, 2] * scale_x   # Principal point in X
f_y = camera_matrix[1, 1] * scale_y  # Focal length in Y
c_y = camera_matrix[1, 2] * scale_y   # Principal point in Y

video_path = "./computer_vision/images/dual_tag_test.mp4"
# video_path = "./computer_vision/images/IMG_6434.MOV"
cap = cv2.VideoCapture(video_path)
output_path = './computer_vision/images/apriltagStandard41h12_test_output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
out = cv2.VideoWriter(output_path, fourcc, 30.0, (640, 360))

center_x_vals = []
center_y_vals = []
width_vals = []
height_vals = []
depth_vals = []  # New depth data
tag_data = {}
tag_size = 0.07  # Example tag size in meters

object_points = array([
    [-tag_size / 2, -tag_size / 2, 0],
    [ tag_size / 2, -tag_size / 2, 0],
    [ tag_size / 2,  tag_size / 2, 0],
    [-tag_size / 2,  tag_size / 2, 0]
], dtype=float32)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None)
    frame = cv2.resize(frame, (640, 360))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    detections1 = detector1.detect(gray_frame)
    detections2 = detector2.detect(gray_frame)

    # Combine results
    detections = detections1 + detections2
    # detections = detector.detect(gray_frame)
    print(len(detections))
    for detection in detections:
        tag_id = detection.tag_id  # Get the tag's ID
        print(tag_id)
        image_points = array(detection.corners, dtype=float32)

        x0, y0 = map(int, detection.corners[0])
        x1, y1 = map(int, detection.corners[1])
        x2, y2 = map(int, detection.corners[2])
        x3, y3 = map(int, detection.corners[3])

        # Pose estimation
        success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, 
                                        flags=cv2.SOLVEPNP_IPPE_SQUARE)
        depth = tvec[2][0] if success else 0  # Depth from translation vector

        # Compute corrected X and Y coordinates
        center_xs = int(((mean(image_points[:, 0]) - c_x) * depth / f_x) + c_x)
        center_ys = int(((mean(image_points[:, 1]) - c_y) * depth / f_y) + c_y)
        
        # Compute bounding box information
        center_x = int((min(x0, x3) + max(x1, x2)) / 2)
        center_y = int((min(y0, y1) + max(y2, y3)) / 2)
        width = int(max(x1, x2) - min(x0, x3))
        height = int(max(y2, y3) - min(y0, y1))
        
        # Correct width and height based on tag size and depth
        widths = int(tag_size * f_x / depth) if depth > 0 else width
        heights = int(tag_size * f_y / depth) if depth > 0 else height

        # Ensure storage for this tag ID
        if tag_id not in tag_data:
            tag_data[tag_id] = {"center_x_vals": [], "center_y_vals": [], 
                                "width_vals": [], "height_vals": [], "depth_vals": []}
        
        # Store values for this tag ID
        tag_data[tag_id]["center_x_vals"].append(center_xs)
        tag_data[tag_id]["center_y_vals"].append(center_ys)
        tag_data[tag_id]["width_vals"].append(widths)
        tag_data[tag_id]["height_vals"].append(heights)
        tag_data[tag_id]["depth_vals"].append(depth)

        # Draw the detection
        for i in range(4):
            pt1 = tuple(round(image_points[i]).astype(int))
            pt2 = tuple(round(image_points[(i + 1) % 4]).astype(int))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)  # Green box

        # Label the tag properly
        cv2.putText(frame, f"ID: {tag_id}", (x0, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Center: ({center_x}, {center_y})", (center_x, center_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, f"W: {width} H: {height} D: {depth:.2f}m", (center_x, center_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    
    # for detection in detections:
    #     image_points = array(detection.corners, dtype=float32)

    #     x0, y0 = map(int, detection.corners[0])
    #     x1, y1 = map(int, detection.corners[1])
    #     x2, y2 = map(int, detection.corners[2])
    #     x3, y3 = map(int, detection.corners[3])
        
    #     # Pose estimation
    #     success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, 
    #                                        flags=cv2.SOLVEPNP_IPPE_SQUARE)
    #     depth = tvec[2][0] if success else 0  # Depth from translation vector
    #     depth_vals.append(depth)
        
    #     # Correct X and Y coordinates using depth and tag size
    #     center_xs = int(((mean(image_points[:, 0]) - c_x) * depth / f_x) + c_x)
    #     center_ys = int(((mean(image_points[:, 1]) - c_y) * depth / f_y) + c_y)
        
    #     center_x = int((min(x0, x3) + max(x1, x2)) / 2)
    #     center_y = int((min(y0, y1) + max(y2, y3)) / 2)
    #     width = int(max(x1, x2) - min(x0, x3))
    #     height = int(max(y2, y3) - min(y0, y1))
    #     # Correct width and height based on tag size and depth
    #     widths = int(tag_size * f_x / depth)
    #     heights = int(tag_size * f_y / depth)

    #     center_x_vals.append(center_xs)
    #     center_y_vals.append(center_ys)
    #     width_vals.append(widths)
    #     height_vals.append(heights)

    #     # Drawing tags
    #     for i in range(4):
    #         pt1 = tuple(round(image_points[i]).astype(int))
    #         pt2 = tuple(round(image_points[(i + 1) % 4]).astype(int))
    #         cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    #     tag_id = str(detection.tag_id)
    #     cv2.putText(frame, f"ID: {tag_id}", (x0, y0 - 10),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #     cv2.putText(frame, f"Center: ({center_x}, {center_y})", (center_x, center_y + 20),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    #     cv2.putText(frame, f"W: {width} H: {height} D: {depth:.2f}m", (center_x, center_y + 40),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    out.write(frame)
    cv2.imshow('AprilTag Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video saved to {output_path}")

# Plotting collected data
# plt.figure(figsize=(12, 10))
# plt.subplot(3, 2, 1)
# plt.plot(center_x_vals, label='Center X')
# plt.title('Center X values')
# plt.legend()

# plt.subplot(3, 2, 2)
# plt.plot(center_y_vals, label='Center Y')
# plt.title('Center Y values')
# plt.legend()

# plt.subplot(3, 2, 3)
# plt.plot(depth_vals, label='Depth (Z)')
# plt.title('Depth (Z) values')
# plt.legend()

# plt.subplot(3, 2, 4)
# plt.plot(width_vals, label='Width')
# plt.title('Width values')
# plt.legend()

# plt.subplot(3, 2, 5)
# plt.plot(height_vals, label='Height')
# plt.title('Height values')
# plt.legend()

# plt.tight_layout()
# plt.show()

for tag_id, data in tag_data.items():
    plt.figure(figsize=(12, 10))  # Create a new figure for each tag

    plt.subplot(3, 2, 1)
    plt.plot(data["center_x_vals"], label=f'Tag {tag_id} - Center X', color='b')
    plt.title(f'Tag {tag_id} - Center X')
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(data["center_y_vals"], label=f'Tag {tag_id} - Center Y', color='g')
    plt.title(f'Tag {tag_id} - Center Y')
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(data["depth_vals"], label=f'Tag {tag_id} - Depth (Z)', color='r')
    plt.title(f'Tag {tag_id} - Depth')
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(data["width_vals"], label=f'Tag {tag_id} - Width', color='m')
    plt.title(f'Tag {tag_id} - Width')
    plt.legend()

    plt.subplot(3, 2, 5)
    plt.plot(data["height_vals"], label=f'Tag {tag_id} - Height', color='c')
    plt.title(f'Tag {tag_id} - Height')
    plt.legend()

    plt.subplot(3, 2, 6)
    plt.plot(data["depth_vals"], label=f'Tag {tag_id} - Depth Repeated', color='orange')
    plt.title(f'Tag {tag_id} - Depth (Repeated)')
    plt.legend()

    plt.suptitle(f"AprilTag {tag_id} Data Over Time")  # Super title for clarity
    plt.tight_layout()
    plt.show()
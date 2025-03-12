import cv2
import platform
from pupil_apriltags import Detector
from numpy import array, int32
from tools.common import loadConfig
import matplotlib.pyplot as plt

system_name = platform.system()
# detector = Detector(families="tagStandard41h12",
#                     nthreads=1,
#                     quad_decimate=1.0,
#                     quad_sigma=0.0,
#                     refine_edges=1,
#                     decode_sharpening=0.25,
#                     debug=0) 

at_detector = cv2.aruco.DetectorParameters()
aruco_dict = cv2.aruco.Dictionary(cv2.aruco.DICT_APRILTAG_41h12)

calibration_data = loadConfig("./configuration_files/camera_configs/calibration_matrix_dslr.json")
camera_matrix = array(calibration_data["camera_matrix"])
dist_coeffs = array(calibration_data["dist_coeff"])

video_path = "./computer_vision/images/apriltagStandard41h12_test.mp4"
cap = cv2.VideoCapture(video_path)
output_path = './computer_vision/images/apriltagStandard41h12_test_output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
out = cv2.VideoWriter(output_path, fourcc, 30.0, (640, 360))

center_x_vals = []
center_y_vals = []
width_vals = []
height_vals = []
depth_vals = []  # New depth data

tag_size = 0.07  # Example tag size in meters
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 360))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = cv2.aruco.detectMarkers(gray_frame, aruco_dict, parameters=at_detector)

    if ids is not None:
        for i in range(len(ids)):
            corner_points = corners[i][0]
            x0, y0 = map(int, corner_points[0])
            x1, y1 = map(int, corner_points[1])
            x2, y2 = map(int, corner_points[2])
            x3, y3 = map(int, corner_points[3])

            center_x = int((x0 + x2) / 2)
            center_y = int((y0 + y2) / 2)
            width = int(max(x1, x2) - min(x0, x3))
            height = int(max(y2, y3) - min(y0, y1))

            # Depth Calculation Placeholder (requires calibration data)
            depth = 0.0  # Replace with actual depth logic if needed

            center_x_vals.append(center_x)
            center_y_vals.append(center_y)
            width_vals.append(width)
            height_vals.append(height)
            depth_vals.append(depth)

            cv2.polylines(frame, [int32(corner_points)], True, (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"ID: {ids[i][0]}", (x0, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(frame)

    cv2.imshow('AprilTag Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video saved to {output_path}")

# Plotting collected data
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.plot(center_x_vals, label='Center X')
plt.title('Center X values')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(center_y_vals, label='Center Y')
plt.title('Center Y values')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(width_vals, label='Width')
plt.title('Width values')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(height_vals, label='Height')
plt.title('Height values')
plt.legend()

plt.tight_layout()
plt.show()
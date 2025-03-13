import cv2 as cv
from numpy import float32, round
from pupil_apriltags import Detector as AprilDetector
from computer_vision.detector import Detector as ObjectDetector
from computer_vision.camera import Camera
from computer_vision.tools.common import *
from computer_vision.tools.stopwatch import Stopwatch
from computer_vision.tools.dataloader import Dataloader

def unpackData(data: dict):
    ids, ts, dts, cxs, cys, widths, heights = [], [], [], [], [], [], []
    for id in sorted(data.keys()):  # Ensure order consistency
        track_ts, track_dts, track_cxs, track_cys, track_widths, track_heights = zip(*data[id])
        ids.append(id)
        ts.append(list(track_ts))
        dts.append(list(track_dts))
        cxs.append(list(track_cxs))
        cys.append(list(track_cys))
        widths.append(list(track_widths))
        heights.append(list(track_heights))
        
    return ts, dts, cxs, cys, widths, heights 
    
class MeasurementGenerator(Camera):
    detector: ObjectDetector
    dataloader: Dataloader
    watch: Stopwatch
    
    def __init__(self, camera_config, detector_config, case_id):
        super().__init__(camera_config, case_id)
        self.detector = ObjectDetector(detector_config)
        self.aprildetector = AprilDetector(families="tagStandard41h12", nthreads=1, quad_decimate=1.0, 
                                           quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25, debug=0)
         
        self.dataloader = Dataloader(detector_config["output"])
        self.watch = Stopwatch()
        
        calibration_data = loadConfig(detector_config["calibration_data"])
        self.camera_matrix = array(calibration_data["camera_matrix"])
        self.dist_coeffs = array(calibration_data["dist_coeff"])
        
        self.tag_size = detector_config["tag_size"]
        self.object_data = {}
        self.tag_data = {}
        self.case_id = case_id
        
        self.idealTag = array([[-detector_config["tag_size"] / 2, -detector_config["tag_size"] / 2],  # Top-left
                               [ detector_config["tag_size"] / 2, -detector_config["tag_size"] / 2],  # Top-right
                               [ detector_config["tag_size"] / 2,  detector_config["tag_size"] / 2],  # Bottom-right
                               [-detector_config["tag_size"] / 2,  detector_config["tag_size"] / 2]   # Bottom-left
                              ], dtype=float32)
        
    def processFrame(self, frame: cv.Mat):
        """This function overrides the processFrame function of class Camera

        Args:
            frame (cv.Mat): the next frame from input feed
        """
        frame = cv.undistort(frame, self.camera_matrix, self.dist_coeffs, None)
        frame = cv.resize(frame, (self.detector.frame_w, self.detector.frame_h))
        april_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # self.objectDetectionProcess(frame)
        self.aprilTagDetectionProcess(april_frame, frame)
        
        if self.write:
            self.output.write(frame)
            
        cv.imshow('Recording', frame)
    
    def objectDetectionProcess(self, frame: cv.Mat):
        detections = self.detector.measurement(frame)
        self.watch.sync()
        if detections:
            for id, obj_data in detections.items():
                cx, cy = obj_data['center']
                width = obj_data['width']
                height = obj_data['height']
                frame = drawDetections(frame, id, cx, cy, width, height)
                
                if id not in self.object_data:
                    self.object_data[id] = []
                
                self.object_data[id].append([self.watch._curr_time, self.watch._dt, cx, cy, width, height])
        else:
            print("No Objects Detected")
            pass
    
    def aprilTagDetectionProcess(self,  april_frame: cv.Mat, frame: cv.Mat):
        focal_length_x = self.camera_matrix[0, 0]  # f_x
        focal_length_y = self.camera_matrix[1, 1]  # f_y
        detections = self.aprildetector.detect(april_frame)
        self.watch.sync()
        for detection in detections:
            tag_id = detection.tag_id  # Get the tag's ID
            image_points = array(detection.corners, dtype=float32)
            
            x0, y0 = map(float, detection.corners[0])
            x1, y1 = map(float, detection.corners[1])
            x2, y2 = map(float, detection.corners[2])
            x3, y3 = map(float, detection.corners[3])
            
            # Compute bounding box
            x_min, x_max = min(x0, x1, x2, x3), max(x0, x1, x2, x3)
            y_min, y_max = min(y0, y1, y2, y3), max(y0, y1, y2, y3)

            # Calculate center and dimensions
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min

            H = detection.homography
            Z0 = 1.5
            
            pixel_coord = array([[cx], [cy], [1]])
            transformed = H @ pixel_coord
            w = transformed[2, 0]  # Extract w (scale factor)
            
            if w != 0:  # Avoid division by zero
                depth = Z0 / w
                real_cx = transformed[0, 0] / w
                real_cy = transformed[1, 0] / w

                # Scale width and height to real-world size
                width_meters = (width / focal_length_x) * depth
                height_meters = (height / focal_length_y) * depth
            else:
                depth = 0
                real_cx, real_cy = 0, 0
                width_meters, height_meters = 0, 0

            
            # Ensure storage for this tag ID
            if tag_id not in self.tag_data:
                self.tag_data[tag_id] = []
            
            # # Store values for this tag ID
            # self.tag_data[tag_id].append([self.watch._curr_time, self.watch._dt, cx, cy, width, height])
            self.tag_data[tag_id].append([self.watch._curr_time, self.watch._dt, real_cx, real_cy, abs(width_meters), abs(height_meters)])

            # Draw the detection
            for i in range(4):
                pt1 = tuple(round(image_points[i]).astype(int))
                pt2 = tuple(round(image_points[(i + 1) % 4]).astype(int))
                cv.line(frame, pt1, pt2, (0, 255, 0), 2)  # Green box

            # Label the tag properly
            cv.putText(frame, f"ID: {tag_id}", (int(cx), int(cy) - 20),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv.putText(frame, f"Center: ({round(real_cx, 4)}, {round(real_cy, 4)})", 
                    (int(cx), int(cy) + 25),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv.putText(frame, f"W: {round(abs(width_meters), 4)} H: {round(abs(height_meters), 4)} D: {round(depth, 4)}", 
                    (int(cx), int(cy) + 45),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    def storeData(self):
        header = ["time", "dt", "Center (x-axis)", "Center (y-axis)", "box width", "box height"]
        data = {}
        
        for obj_id, obj_rows in self.object_data.items():
            data[f'object_{obj_id}'] = obj_rows

        for tag_id, tag_rows in self.tag_data.items():
            data[f'tag_{tag_id}'] = tag_rows

        if not data:
            print("Warning: No data to save.")
        else:
            csv_file = self.dataloader.save(data, header)
        print(f"Data saved in: {csv_file}")
    
    def returnData(self):
        obj_ts, obj_dts, obj_cxs, obj_cys, obj_widths, obj_heights = unpackData(self.object_data)
        tag_ts, tag_dts, tag_cxs, tag_cys, tag_widths, tag_heights = unpackData(self.tag_data)
        
        ts = obj_ts + tag_ts
        dts = obj_dts + tag_dts
        cxs = obj_cxs + tag_cxs
        cys = obj_cys + tag_cys
        widths = obj_widths + tag_widths
        heights = obj_heights + tag_heights

        return ts, dts, cxs, cys, widths, heights

   
    # def returnData(self):
    #     ts = []
    #     dts = []
    #     cxs = []
    #     cys = []
    #     widths = []
    #     heights = []
        
    #     for track_id in sorted(self.object_data.keys()):  # Ensure order consistency
    #         track_ts = []
    #         track_dts = []
    #         track_cxs = []
    #         track_cys = []
    #         track_widths = []
    #         track_heights = []

    #         for row in self.object_data[track_id]:
    #             t, dt, cx, cy, width, height = row
    #             track_ts.append(t)
    #             track_dts.append(dt)
    #             track_cxs.append(cx)
    #             track_cys.append(cy)
    #             track_widths.append(width)
    #             track_heights.append(height)

    #         ts.append(track_ts)
    #         dts.append(track_dts)
    #         cxs.append(track_cxs)
    #         cys.append(track_cys)
    #         widths.append(track_widths)
    #         heights.append(track_heights)
        
    #     return ts, dts, cxs, cys, widths, heights
        
    def defaultMeasurementGenerationProcess(self):
        self.initRecording()
        self.startRecording()
        self.storeData()
        ts, dts, cxs, cys, widths, heights = self.returnData()
            
        return ts, dts, cxs, cys, widths, heights

   
if __name__ == "__main__":
    import sys
    
    def testbenchMeasurementGenerator(camera_config: json, detector_config: json, case_id: str):
        generator = MeasurementGenerator(camera_config, detector_config, case_id)
        ts, dts, cxs, cys, widths, heights = generator.defaultMeasurementGenerationProcess()

        # Comment out if we do not need to print out metrics and additional graphs
        for i in range(len(cys)):
            terminal = sys.stdout
            getDataMetrics(f"{case_id}_point_{i}", cxs[i], cys[i], widths[i], heights[i])
            sys.stdout = open(f"./output/{case_id}_point_{i}_metrics.txt", 'w')
            getDataMetrics(f"{case_id}_point_{i}", cxs[i], cys[i], widths[i], heights[i])
            sys.stdout = terminal
            detectionGraphics(f"{case_id}_point_{i}", ts[i], cxs[i], cys[i], widths[i], heights[i], True)
        plt.show()
        
    # case_id = "m095_0_k80_80"
    # case_id = "noise_test"
    # detector_config = loadConfig('./configuration_files/detector_configs/detector_spring.json')
    
    # case_id = "sport"
    case_id = "sport_load"
    detector_config = loadConfig('./configuration_files/detector_configs/detector_vehicle.json')
    
    # These can remain uncommented
    camera_config = loadConfig(f'./configuration_files/camera_configs/camera_{case_id}.json')
    testbenchMeasurementGenerator(camera_config, detector_config, case_id)
    
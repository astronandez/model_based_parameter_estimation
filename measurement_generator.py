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
        self.frame_index = 0
        
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
        # frame = cv.rotate(frame, cv.ROTATE_180)
        # frame = cv.undistort(frame, self.camera_matrix, self.dist_coeffs, None)
        frame = cv.resize(frame, (self.detector.frame_w, self.detector.frame_h))
        april_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        self.objectDetectionProcess(frame)
        # self.aprilTagDetectionProcess(april_frame, frame)
        
        if self.write:
            self.output.write(frame)
            
        cv.imshow('Recording', frame)
        self.frame_index += 1  # Track frame count
    
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
                
                # self.object_data[id].append([self.watch._curr_time, self.watch._dt, cx, cy, width, height])
                elapsed_time = self.frame_index / self.fps  

                # Append the calculated time
                self.object_data[id].append([elapsed_time, 1 / self.fps, cx, cy, abs(width), abs(height)])
        else:
            print("No Objects Detected")
            pass
    
    def aprilTagDetectionProcess(self,  april_frame: cv.Mat, frame: cv.Mat):
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
            
            # Ensure storage for this tag ID
            if tag_id not in self.tag_data:
                self.tag_data[tag_id] = []
            
            elapsed_time = self.frame_index / self.fps  

            # Append the calculated time
            self.tag_data[tag_id].append([elapsed_time, 1 / self.fps, cx, cy, abs(width), abs(height)])
                        
            # # Store values for this tag ID
            # self.tag_data[tag_id].append([self.watch._curr_time, self.watch._dt, cx, cy, abs(width), abs(height)])

            # Draw the detection
            drawAprilTags(frame, image_points, tag_id, cx, cy, width, height)
                  
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
    
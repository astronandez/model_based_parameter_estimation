import cv2 as cv

from computer_vision.detector import Detector
from computer_vision.camera import Camera
from computer_vision.tools.common import *
from computer_vision.tools.stopwatch import Stopwatch
from computer_vision.tools.dataloader import Dataloader
    
class MeasurementGenerator(Camera):
    detector: Detector
    dataloader: Dataloader
    watch: Stopwatch
    
    def __init__(self, camera_config, detector_config, case_id):
        super().__init__(camera_config, case_id)
        self.detector = Detector(detector_config)
        self.dataloader = Dataloader(detector_config["output"])
        self.watch = Stopwatch()
        self.data = {}
        self.case_id = case_id
        
    def processFrame(self, frame: cv.Mat):
        """This function overrides the processFrame function of class Camera

        Args:
            frame (cv.Mat): the next frame from input feed
        """
        detections = self.detector.measurement(frame)
        self.watch.sync()
        if detections:
            for id, obj_data in detections.items():
                cx, cy = obj_data['center']
                width = obj_data['width']
                height = obj_data['height']
                frame = drawDetections(frame, id, cx, cy, width, height)
                
                if id not in self.data:
                    self.data[id] = []
                
                self.data[id].append([self.watch._curr_time, self.watch._dt, cx, cy, width, height])
                
                # Retaining code below for more information when desired
                # print(f"t: {self.watch._curr_time}, dt: {self.watch._dt}" )
                # print(f"Track ID: {track_id}, Data: { {k: float(v) if isinstance(v, float32) 
                #       else (tuple(float(x) for x in v) if isinstance(v, tuple) else v) for k, v in obj_data.items()} }")
        else:
            print("No measurement recorded")
            pass
        
        cv.imshow('Recording', frame)
        if self.write:
            self.output.write(frame) 
    
    def storeData(self):
        header = ["time", "dt", "Center (x-axis)", "Center (y-axis)", "box width", "box height"]
        csv_file = self.dataloader.save(self.data, header)
        print(f"Data saved in: {csv_file}")
        
    def returnData(self):
        ts = []
        dts = []
        cxs = []
        cys = []
        widths = []
        heights = []
        
        for track_id in sorted(self.data.keys()):  # Ensure order consistency
            track_ts = []
            track_dts = []
            track_cxs = []
            track_cys = []
            track_widths = []
            track_heights = []

            for row in self.data[track_id]:
                t, dt, cx, cy, width, height = row
                track_ts.append(t)
                track_dts.append(dt)
                track_cxs.append(cx)
                track_cys.append(cy)
                track_widths.append(width)
                track_heights.append(height)

            ts.append(track_ts)
            dts.append(track_dts)
            cxs.append(track_cxs)
            cys.append(track_cys)
            widths.append(track_widths)
            heights.append(track_heights)
        
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
    
    case_id = "sport"
    case_id = "sport_load"
    detector_config = loadConfig('./configuration_files/detector_configs/detector_vehicle.json')
    
    # These can remain uncommented
    camera_config = loadConfig(f'./configuration_files/camera_configs/camera_{case_id}.json')
    testbenchMeasurementGenerator(camera_config, detector_config, case_id)
    
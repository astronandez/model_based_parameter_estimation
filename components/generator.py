import json
import cv2 as cv
from numpy import float32

from components.detector import Detector
from camera import Camera
from tools.common import *
from tools.stopwatch import Stopwatch
from components.tools.dataloader import Dataloader

def drawDetections(frame, cx, cy, width, height):
    x1, y1, x2, y2 = centerToBoundingBox(cx, cy, width, height)
    cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv.putText(frame, f'z: {cy} px', (int(cx), int(cy) - int(height/2 + 5)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

class Generator(Camera):
    detector: Detector
    dataloader: Dataloader
    watch: Stopwatch
    
    def __init__(self, camera_config, detector_config):
        super().__init__(camera_config)
        self.detector = Detector(detector_config)
        self.watch = Stopwatch()
        self.dataloader = Dataloader("./output/")
        self.data = {}
        
    def processFrame(self, frame):
        detections = self.detector.measurement(frame)
        self.watch.sync()
        if detections:
            for i in range(len(detections)):
                for track_id, obj_data in detections.items():
                    center = obj_data['center']
                    cx, cy = center
                    width = obj_data['width']
                    height = obj_data['height']
                    frame = drawDetections(frame, cx, cy, width, height)
                    
                    if track_id not in self.data:
                        self.data[track_id] = []
                    
                    self.data[track_id].append([self.watch._curr_time, self.watch._dt, cx, cy, width, height])
                    
                    # Retaining code below for more information as desired
                    # print(f"t: {self.watch._curr_time}, dt: {self.watch._dt}" )
                    # print(f"Track ID: {track_id}, Data: { {k: float(v) if isinstance(v, float32) 
                    #       else (tuple(float(x) for x in v) if isinstance(v, tuple) else v) for k, v in obj_data.items()} }")
        else:
            print("Nothing found >:(")
        
        cv.imshow('Recording', frame) 
    
    def storeData(self):
        header = ["time", "dt", "Center (x-axis)", "Center (y-axis)", "box width", "box height"]
        self.dataloader.save(self.data, header)
    
if __name__ == "__main__":
    camera_config_path = './configuration_files/camera_configs/camera_generator.json'
    detector_config_path = './configuration_files/detector_configs/detector_base.json'
    
    camera_config = loadConfig(camera_config_path)
    detector_config = loadConfig(detector_config_path)
    
    generator = Generator(camera_config, detector_config)
    generator.initRecording()
    generator.startRecording()
    generator.storeData()
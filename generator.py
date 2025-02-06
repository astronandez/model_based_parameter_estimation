import json
import cv2 as cv
from numpy import float32

from components.detector import Detector
from components.camera import Camera
from components.tools.common import *
from components.tools.stopwatch import Stopwatch
from components.tools.dataloader import Dataloader

def defaultMeasurementGeneration(camera_config, detector_config):
    generator = Generator(camera_config, detector_config)
    generator.initRecording()
    generator.startRecording()
    ts, dts, cxs, cys = generator.storeData()
            
    return ts, dts, cxs, cys
    
class Generator(Camera):
    detector: Detector
    dataloader: Dataloader
    watch: Stopwatch
    
    def __init__(self, camera_config, detector_config):
        super().__init__(camera_config)
        self.detector = Detector(detector_config)
        self.watch = Stopwatch()
        self.dataloader = Dataloader(detector_config["output"])
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
                    
                    # Retaining code below for more information when desired
                    # print(f"t: {self.watch._curr_time}, dt: {self.watch._dt}" )
                    # print(f"Track ID: {track_id}, Data: { {k: float(v) if isinstance(v, float32) 
                    #       else (tuple(float(x) for x in v) if isinstance(v, tuple) else v) for k, v in obj_data.items()} }")
        else:
            print("Nothing found >:(")
        
        cv.imshow('Recording', frame)
        if self.write:
            self.output.write(frame) 
    
    def storeData(self):
        header = ["time", "dt", "Center (x-axis)", "Center (y-axis)", "box width", "box height"]
        self.dataloader.save(self.data, header)
        
        ts = []
        dts = []
        cxs = []
        cys = []

        for track_id in self.data:
            for row in self.data[track_id]:
                t, dt, cx, cy, width, height = row
                ts.append(t)
                dts.append(dt)
                cxs.append(cx)
                cys.append(cy)
                
        return ts, dts, cxs, cys
        
    
if __name__ == "__main__":
    camera_config_path = './configuration_files/camera_configs/camera_generator.json'
    detector_config_path = './configuration_files/detector_configs/detector_spring.json'
    camera_config = loadConfig(camera_config_path)
    detector_config = loadConfig(detector_config_path)
    
    model_id = "m105_7_k82_3448"
    ts, dts, cxs, cys = defaultMeasurementGeneration(camera_config, detector_config)
    inspectData(model_id, ts, dts, cxs, cys)
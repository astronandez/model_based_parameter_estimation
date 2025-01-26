import os
import json
import cv2 as cv
from numpy import array, argmax
os.environ['YOLO_VERBOSE'] = 'False'
from ultralytics import YOLO

class Detector:
    config: json
    detector: YOLO
    
    def __init__(self, config: json):
        self.config = config
        self.detector = YOLO(config['detector'], verbose=False)
        self.fw = config['screen_width']
        self.fh = config['screen_height']
        
    def reading(self, frame: cv.Mat):
        results = self.detector.track(frame, conf=0.6, iou=0.5, imgsz=(self.fw, self.fh), persist=True)
        detections = results[0].boxes.xywh.numpy()
        objs = []

        if len(detections) > 0:
            track_ids = results[0].boxes.id
            prob_class = results[0].probs
            if track_ids is not None:
                for i in range(len(detections)):
                    cx, cy, width, height = detections[i]
                    data = array([int(track_ids[i]), results[0].names[argmax(prob_class)], cx, cy, width, height], dtype=object)
                    objs.append(data)
                
            return objs
        else:
            return None

        
############ Testbench ############    
if __name__ == "__main__":
    def loadConfig(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    
    config_path = './configuration_methods/detector_config.json'
    config = loadConfig(config_path)
    cam = Detector(config)
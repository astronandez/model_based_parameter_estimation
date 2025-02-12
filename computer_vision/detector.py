import os
import json
import cv2 as cv
from numpy import array, argmax
os.environ['YOLO_VERBOSE'] = 'False'
from ultralytics import YOLO
from computer_vision.tracker import Tracker
from computer_vision.tools.common import loadConfig

class Detector:
    model: YOLO
    frame_w: int
    frame_h: int
    
    def __init__(self, config):
        self.detector = YOLO(config['detector'], verbose=False)
        self.tracker  = Tracker(max_age=2)
        self.frame_w = config['frame_w']
        self.frame_h = config['frame_h']
    
    def measurement(self, frame: cv.Mat):
        detections = self.detector.track(frame, conf=0.6, iou=0.5, imgsz=(self.frame_w, self.frame_h), persist=False)
        ids = detections[0].boxes.id
        
        if ids is not None:
            objs = self.tracker.newDetections(detections)
            return objs
        else:
            return None    
        
if __name__ == "__main__":
    config_path = './configuration_files/detector_configs/detector_base.json'
    config = loadConfig(config_path)
    cam = Detector(config)
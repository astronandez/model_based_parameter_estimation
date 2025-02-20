import os
import json
import cv2 as cv
from numpy import array, argmax
from numpy.linalg import norm
os.environ['YOLO_VERBOSE'] = 'False'
from ultralytics import YOLO
from computer_vision.tools.common import loadConfig  

class Detector:
    model: YOLO
    frame_w: int
    frame_h: int
    
    def __init__(self, config):
        self.detector = YOLO(config['detector'], verbose=False)
        self.frame_w = config['frame_w']
        self.frame_h = config['frame_h']
        self.objects = {}  # Tracked objects
        self.max_age = config['max_age']  # Allow objects to persist longer
        self.min_dist = config['min_dist'] # The minimum distance allowed between centroids when tracking
        self.next_id = 0  # Counter for assigning new IDs

    def measurement(self, frame: cv.Mat):
        detections = self.detector.predict(frame, conf=0.5, iou=0.5, imgsz=(self.frame_w, self.frame_h))  

        # If no detections are found we need to update our set of current tracks and return None
        if not detections or detections[0].boxes is None:
            # print("No Detection found in frame")
            self.incrementObjectAge(set())
            self.removeOldObjects()
            return self.objects if self.objects else None

        boxes = detections[0].boxes.xywh.cpu().numpy()
        new_ids = set()
        new_obj = {}

        for cx, cy, width, height in boxes:
            closest_id = self.findClosestCentroid(cx, cy)
            
            if closest_id is not None:
                new_ids.add(closest_id)
            else:
                closest_id = self.next_id
                self.next_id += 1

            new_obj[closest_id] = {
                'center': (cx, cy),
                'width': width,
                'height': height,
                'age': 0
            }

        self.objects = new_obj
        self.incrementObjectAge(new_ids)
        self.removeOldObjects()
        
        return self.objects if self.objects else None
    
    def findClosestCentroid(self, cx, cy):
        closest_id = None
        closest_centroid = self.min_dist
        for id, obj in self.objects.items():
            dist = norm(array(obj['center']) - array((cx, cy)))
            if dist < closest_centroid:  # Find closest match
                closest_centroid = dist
                closest_id = id
                
        return closest_id
            
    def removeOldObjects(self):
        for id in list(self.objects.keys()):
            if self.objects[id]['age'] >= self.max_age:
                del self.objects[id]
                # print(f"Removed inactive Object {id}")

    def incrementObjectAge(self, new_ids):
        for obj_id in self.objects:
            if obj_id not in new_ids:  # Only increment if the object was not updated
                self.objects[obj_id]['age'] += 1
                       
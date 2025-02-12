from numpy import array, argmax
from ultralytics.engine.results import Results

class Tracker:
    objects: dict
    
    def __init__(self, max_age):
        self.max_age = max_age
        self.objects = {}
        
    def enterObject(self, track_id: int, class_name, cx: float, cy: float, width: float, height: float):
        """This function is used to add or update an object in the list of tracker.objects
        
        Args:
            track_id (int): The id of the object
            class_name (_type_): Class name of detected object
            cx (float): The center position in the x-axis of the object in frame 
            cy (float): The center position in the y-axis of the object in frame 
            width (float): The width of the object in frame
            height (float): The height of the object in frame
        """
        self.objects[int(track_id)] = {
            'class_name': class_name,
            'center': (cx, cy),
            'width': width,
            'height': height,
            'age': 0
        }
        
    def removeObject(self, track_id: int):
        """This function is used to remove objects from the tracker.objects dictionary

        Args:
            track_id (int): The id matching the desired object to be removed
        """
        
        if track_id in self.objects:
            del self.objects[track_id]
    
    def incrementObjectAge(self):
        for track_ids in self.objects:
            self.objects[track_ids]['age'] += 1
        
    def removeStaleObjects(self):
        for track_ids in list(self.objects.keys()):
            if self.objects[track_ids]['age'] > self.max_age:
                self.removeObject(track_ids)
                print(f"Removed inactive Object {track_ids}")
               
    def updateObjects(self, ids, names, boxes, prob_class):
        for i in range(len(boxes)):
            cx, cy, width, height = boxes[i]
            id = int(ids[i])
            name = names[argmax(prob_class)]           
            self.enterObject(id, name, cx, cy, width, height)
                   
    def newDetections(self, detections: list[Results]):
        ids = detections[0].boxes.id
        boxes = detections[0].boxes.xywh.numpy()
        names = detections[0].names
        prob_class = detections[0].probs

        self.incrementObjectAge()
        self.updateObjects(ids, names, boxes, prob_class)
        self.removeStaleObjects()
        return self.objects
    
if __name__ == "__main__":
    pass
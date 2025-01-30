import json
import cv2 as cv
from numpy import array
from components.computer_vision.detector import Detector
from components.computer_vision.camera import Camera
from components.tools.stopwatch import Stopwatch
from components.tools.common import *

def drawTireDetections(frame, points1, points2, track_id, cx, cy, width, height):
    x1, y1, x2, y2 = centerToBoundingBox(cx, cy, width, height)
    if track_id == 1:
        points1.append((int(cx), int(cy)))
        cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv.putText(frame, f'z: {cy} px', (int(cx), int(cy) - int(height/2 + 5)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        points2.append((int(cx), int(cy)))
        cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv.putText(frame, f'z: {cy} px', (int(cx), int(cy) - int(height/2 + 5)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    for j in range(len(points1)):
        cv.circle(frame, points1[j], radius=2, color=(0, 0, 255), thickness=-1)
        
    for k in range(len(points2)):
        cv.circle(frame, points2[k], radius=2, color=(0, 255, 0), thickness=-1)

class Record:
    def __init__(self, config: json):
        self.camera = Camera(config)
        self.stopwatch = Stopwatch()
        self.detector = Detector(config)
        self.dataloader = Dataloader()
        self.config = config
        self.start()
             
    def start(self):
        t = 0
        t_data = []
        dt_data = []
        cx_data = []
        cy_data = []
        height_data = []
        width_data = []
        
        t_data2 = []
        dt_data2 = []
        cx_data2 = []
        cy_data2 = []
        height_data2 = []
        width_data2 = []
        
        upper_1 = []
        upper_2 = []
        lower_1 = []
        lower_2 = []
        
        green = (0, 255, 0)
        red = (0, 0, 255)

        while self.camera.cap.isOpened():
            ret, frame = self.camera.cap.read()
            if ret:
                if self.config['rotate']:
                    frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
                    
                if self.config['run']:
                    self.stopwatch.sync()
                    data = self.detector.reading(frame)
                    if data:
                        for i in range(len(data)):
                            track_id = data[i][0]
                            cls = data[i][1]
                            cx = data[i][2]
                            cy = data[i][3]
                            width = data[i][4]
                            height = data[i][5]
                            print(track_id)
                            if self.config['display']:
                                x1, y1, x2, y2 = centerToBoundingBox(cx, cy, width, height)
                                if track_id == 1:
                                    upper_1.append((int(cx), int(cy)))
                                    cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), red, 2)
                                    cv.putText(frame, f'z: {cy} px', (int(cx), int(cy) - int(height/2 + 5)), cv.FONT_HERSHEY_SIMPLEX, 0.5, red, 2)
                                elif track_id == 3:
                                    lower_1.append((int(cx), int(cy)))
                                    cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), red, 2)
                                    cv.putText(frame, f'z: {cy} px', (int(cx), int(cy) - int(height/2 + 5)), cv.FONT_HERSHEY_SIMPLEX, 0.5, red, 2)
                                elif track_id == 9:
                                    upper_2.append((int(cx), int(cy)))
                                    cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), green, 2)
                                    cv.putText(frame, f'z: {cy} px', (int(cx), int(cy) - int(height/2 + 5)), cv.FONT_HERSHEY_SIMPLEX, 0.5, green, 2)
                                    t_data.append(self.stopwatch.curr_time)
                                    dt_data.append(self.stopwatch.dt)
                                    cx_data.append(cx)
                                    cy_data.append(cy)
                                    width_data.append(width)
                                    height_data.append(height)
                                elif track_id == 10:
                                    lower_2.append((int(cx), int(cy)))
                                    cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), green, 2)
                                    cv.putText(frame, f'z: {cy} px', (int(cx), int(cy) - int(height/2 + 5)), cv.FONT_HERSHEY_SIMPLEX, 0.5, green, 2)
                                    t_data2.append(self.stopwatch.curr_time)
                                    dt_data2.append(self.stopwatch.dt)
                                    cx_data2.append(cx)
                                    cy_data2.append(cy)
                                    width_data2.append(width)
                                    height_data2.append(height)
        
                                for j in range(len(upper_1)):
                                    cv.circle(frame, upper_1[j], radius=2, color= red, thickness=-1)
                                    
                                for j in range(len(lower_1)):
                                    cv.circle(frame, lower_1[j], radius=2, color= red, thickness=-1)
                                
                                for k in range(len(upper_2)):
                                    cv.circle(frame, upper_2[k], radius=2, color= green, thickness=-1)
                                    
                                for k in range(len(lower_2)):
                                    cv.circle(frame, lower_2[k], radius=2, color= green, thickness=-1)
                           
                self.camera.cap_out.write(frame)
                cv.imshow('Sensor', frame)
                if cv.waitKey(1) & 0xFF == ord("q"):
                    self.camera.cap.release()
                    self.camera.cap_out.release()
                    cv.destroyAllWindows()
                    break
            else:
                break
    
        full_data = zip(list(range(0, len(t_data))), (array(t_data) - t_data[0]).tolist(), dt_data, cx_data, cy_data, width_data, height_data)
        header = ["index", "time", "dt", "Center (x-axis)", "Center (y-axis)", "box width", "box height"]
        self.dataloader.storeData(full_data, header, f"{self.config['measurements_output'][:-4]}_lower{self.config['measurements_output'][-4:]}")
        
        full_data = zip(list(range(0, len(t_data2))), (array(t_data2) - t_data2[0]).tolist(), dt_data2, cx_data2, cy_data2, width_data2, height_data2)
        header = ["index", "time", "dt", "Center (x-axis)", "Center (y-axis)", "box width", "box height"]
        self.dataloader.storeData(full_data, header, f"{self.config['measurements_output'][:-4]}_upper{self.config['measurements_output'][-4:]}")
                    
if __name__ == "__main__":
    # config_path1 = './configurations/lemur_sport1.json'
    # config_path2 = './configurations/lemur_sport2.json'
    # config_path3 = './configurations/lemur_sport3.json'
    # config_path4 = './configurations/lemur_suv1.json'
    # config_path5 = './configurations/lemur_suv2.json'
    # config_path6 = './configurations/lemur_suv3.json'
    # config_path7 = './configurations/lemur_sport_full1.json'
    # config_path8 = './configurations/lemur_sport_full2.json'
    # config_path9 = './configurations/lemur_sport_full3.json'
    # config_path10 = './configurations/lemur_suv_twopass1.json'
    # config_path11 = './configurations/lemur_suv_twopass2.json'
    # config_path12 = './configurations/lemur_suv_twopass3.json'
    
    # config_list = [config_path1, config_path2, config_path3, config_path4, config_path5, config_path6,
    #                config_path7, config_path8, config_path9, config_path10, config_path11, config_path12]
    
    config_list = ['./configurations/lemur_sticky_sport_load.json']
    for i in config_list:
        config = loadConfig(i)
        record = Record(config)

import json
import cv2 as cv
from seaborn import histplot
from scipy.stats import norm

from computer_vision.detector import Detector
from computer_vision.camera import Camera
from computer_vision.tools.common import *
from computer_vision.tools.stopwatch import Stopwatch
from computer_vision.tools.dataloader import Dataloader
    
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
        widths = []
        heights = []

        for track_id in self.data:
            for row in self.data[track_id]:
                t, dt, cx, cy, width, height = row
                ts.append(t)
                dts.append(dt)
                cxs.append(cx)
                cys.append(cy)
                widths.append(width)
                heights.append(height)
                
        return ts, dts, cxs, cys, widths, heights
        
def defaultMeasurementGeneration(camera_config, detector_config):
    generator = Generator(camera_config, detector_config)
    generator.initRecording()
    generator.startRecording()
    ts, dts, cxs, cys, widths, heights = generator.storeData()
            
    return ts, dts, cxs, cys, widths, heights

   
if __name__ == "__main__":
    import sys
    model_id = "m95_0_k80_80"
    camera_config = loadConfig('./configuration_files/camera_configs/camera_generator.json')
    detector_config = loadConfig('./configuration_files/detector_configs/detector_base.json')
    # ts, dts, cxs, cys, widths, heights = defaultMeasurementGeneration(camera_config, detector_config)

    dataloader = Dataloader("./output/")
    ts, dts, cxs, cys, widths, heights = dataloader.load(f"./data/{model_id}.csv")
    inspectData(camera_config['model_id'], ts, dts, cxs, cys, widths, heights)
    sys.stdout = open(f"./output/{model_id}_metrics.txt", 'w')
    mean_v = mean(cys)
    var_v = var(cys)
    stdd_v = std(cys)
    
    mean_w = mean(cxs)
    var_w = var(cxs)
    stdd_w = std(cxs)
    
    print(f"Model: {model_id}")
    print("===== Measurement Noise Metrics =====")
    print("Mean of y:", mean_v) 
    print("Variance of y:", var_v)
    print("Standard Deviation of y:", stdd_v)
    print("======== System Noise Metrics =======")
    print("Mean of x:", mean_w) 
    print("Variance of x:", var_w)
    print("Standard Deviation of x:", stdd_w, "\n")   
    sys.stdout.close()
    
    label_cys = [f"./graphs/{model_id}_cys_distribution.fig",
                    f'Probability Distribution {model_id} y-axis center (Mean = 0)',
                    "Position (px)",
                    "Probability Density"]
    
    label_cxs = [f"./graphs/{model_id}_cxs_distribution.fig",
                f'Probability Distribution {model_id} x-axis center (Mean = 0)',
                "Position (px)",
                "Probability Density"]
    
    plotDistribution(cys, label_cys)
    plotDistribution(cxs, label_cxs)
    plt.show()
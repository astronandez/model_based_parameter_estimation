import os
import sys

from generator import Generator
from fitter import fitToRealData
from evaluation import Evaluation, defaultEvaluation, default2DEvaluation
from computer_vision.tools.common import *

def getFilePath():
    while True:
        file_path = input("Enter Data file path: ")
        if os.path.isfile(file_path):
            print(f"File found at: {file_path}")
            return file_path
        else:
            print("Invalid path or file does not exist. Please try again.")

def getDataMetrics(model_id, cxs, cys, widths, heights):
        cy_mean, cy_var, cy_std = defaultMetrics(cys)
        cx_mean, cx_var, cx_std = defaultMetrics(cxs)
        wid_mean, wid_var, wid_std = defaultMetrics(widths)
        h_mean, h_var, h_std = defaultMetrics(heights)
            
        print(f"Model Metrics: {model_id}")
        print("=====================================")
        print("Mean of center in y-axis:", cy_mean) 
        print("Variance of center in y-axis:", cy_var)
        print("Standard Deviation of center in y-axis:", cy_std)
        print("=====================================")
        print("Mean of center in x-axis:", cx_mean) 
        print("Variance of center in x-axis:", cx_var)
        print("Standard Deviation of center in x-axis:", cx_std)
        print("=====================================")
        print("Mean of detection width:", wid_mean) 
        print("Variance of detection width:", wid_var)
        print("Standard Deviation of detection width:", wid_std)
        print("=====================================")
        print("Mean of detection height:", h_mean) 
        print("Variance of detection height:", h_var)
        print("Standard Deviation of detection height:", h_std) 

def getParams(prompt, default_values=None):
    print(f"\n{prompt}")
    if default_values is not None:
        print(f"Press Enter to use the default values: {default_values}")
        
    user_input = input(f"Enter values separated by commas: ")
    
    if user_input.strip() == "":
        return default_values
    else:
        try:
            values = [float(x.strip()) for x in user_input.split(",")]
            if len(values) != len(default_values):
                raise ValueError(f"Please enter exactly 6 values.")
            return values
        except ValueError as e:
            print(f"Invalid input: {e}")
            return getParams(prompt, default_values)  # Retry on error     

class Harness:
    generator: Generator
    evaluation: Evaluation
    
    def __init__(self, harness_config_path):
        harness_config = loadConfig(harness_config_path)
        evaluation_config = loadConfig(harness_config["evaluation_config_path"])
        camera_config = loadConfig(harness_config["camera_config_path"])
        detector_config = loadConfig(harness_config["detector_config_path"])
        
        self.generator = Generator(camera_config, detector_config)
        self.evaluation = Evaluation(evaluation_config)
        
        self.from_recording = harness_config['from_recording']
        self.store_metrics = harness_config['store_metrics']
        self.store_graphs = harness_config['store_graphs']
        self.create_fit = harness_config['create_fit']
        self.highdim = harness_config['highdim']
        
        self.evaluation_config = evaluation_config
        self.camera_config = camera_config
        self.detector_config = detector_config
         
    def test(self):    
        # If true, we're creating a new dataset from a video file, else we're reading an existing dataset from a file
        if self.from_recording:
            self.generator.initRecording()
            self.generator.startRecording()
            ts, dts, cxs, cys, widths, heights = self.generator.storeData()
        else:
            file_path = getFilePath()
            ts, dts, cxs, cys, widths, heights = self.generator.dataloader.load(file_path)
        
        # If true, we store the metrics from the dataset in a .txt file, else we display our metrics to terminal only    
        if self.store_metrics:
            terminal = sys.stdout
            getDataMetrics(self.evaluation_config['model_id'], cxs, cys, widths, heights)
            sys.stdout = open(f"./output/{self.evaluation_config['model_id']}_metrics.txt", 'w')
            getDataMetrics(self.evaluation_config['model_id'], cxs, cys, widths, heights)
            sys.stdout = terminal
        else:
            getDataMetrics(self.evaluation_config['model_id'], cxs, cys, widths, heights)
        
        # Create graphs for the timeseries and plot their distributions
        inspectData(self.evaluation_config['model_id'], ts, cxs, cys, widths, heights, self.store_graphs)
        print("Graphs created for time-series data and their respective distributions")
        
        # If true, we want to fit a synthetic function to the actual dataset, else pass
        if self.create_fit:
            params = getParams(f"Provide the true values of dataset: {self.evaluation_config['model_id']}  m, k, b, amplitude, phi, offset:")
            params_guess = getParams("Provide initial guess of the fitment function m, k, b, amplitude, phi, offset:", [0.01, 10.0, 0.001, 10, 3.14, 0.0])
            fitToRealData(self.evaluation_config['model_id'], params, params_guess, ts, cys, self.store_graphs)
        
        if self.highdim:
            default2DEvaluation(self.evaluation, self.evaluation_config, ts, dts, cxs, cys, store=self.store_graphs)
        else:
            defaultEvaluation(self.evaluation, self.evaluation_config, ts, dts, cys, store=self.store_graphs)
        
        plt.show()      
        
if __name__ == "__main__":
    harness_config_path = "./configuration_files/harness_config.json" 
    harness = Harness(harness_config_path)
    harness.test()   
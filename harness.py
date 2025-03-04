import os
import sys

from measurement_generator import MeasurementGenerator
from fitter import defaultFitment
from evaluation import Evaluation
from computer_vision.tools.common import *

def getFilePath():
    while True:
        file_path = input("Enter Data file path: ")
        if os.path.isfile(file_path):
            print(f"File found at: {file_path}")
            return file_path
        else:
            print("Invalid path or file does not exist. Please try again.")

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

def loadHarnessConfig(config_path):
    """Loads a JSON configuration file and applies overrides if it's the top-level config."""
    with open(config_path, "r") as f:
        config = json.load(f)

    # If this is the top-level config, update lower-level configs dynamically
    if "camera_config_path" in config and "detector_config_path" in config:
        for key in ["camera_config_path", "detector_config_path"]:
            lower_config_path = config[key]

            with open(lower_config_path, "r") as f:
                lower_config = json.load(f)

            # Override values dynamically based on the top-level config
            for k in lower_config.keys():
                if k in config:
                    lower_config[k] = config[k]

            # Save the updated lower-level config file
            with open(lower_config_path, "w") as f:
                json.dump(lower_config, f, indent=4)

    return config  # Return the loaded (and possibly modified) config

class Harness:
    generator: MeasurementGenerator
    evaluation: Evaluation
    
    def __init__(self, harness_config_path):
        harness_config = loadHarnessConfig(harness_config_path)
        evaluation_config = loadConfig(harness_config["evaluation_config_path"])
        camera_config = loadConfig(harness_config["camera_config_path"])
        detector_config = loadConfig(harness_config["detector_config_path"])
        
        self.generator = MeasurementGenerator(camera_config, detector_config, harness_config['case_id'])
        self.evaluation = Evaluation(evaluation_config, harness_config['case_id'])
        self.model_name = evaluation_config['model_name']
        
        self.case_id = harness_config['case_id']
        self.from_recording = harness_config['from_recording']
        self.store_metrics = harness_config['store_metrics']
        self.store_graphs = harness_config['store_graphs']
        self.store_fit = harness_config['store_fit']
        self.store_eval = harness_config['store_eval']

        self.evaluation_config = evaluation_config
        self.camera_config = camera_config
        self.detector_config = detector_config
        
    def sourceMeasurements(self):
        if self.from_recording:
            measurements = self.generator.defaultMeasurementGenerationProcess()
        else:
            file_path = getFilePath()
            measurements = self.generator.dataloader.load(file_path)
        return measurements
    
    def storeMetrics(self, cxs, cys, widths, heights):
        if self.store_metrics:
            terminal = sys.stdout
            getDataMetrics(self.case_id, cxs, cys, widths, heights)
            
            i = 0
            while os.path.exists(f"./output/{self.case_id}_metrics_{i}.txt"):
                i += 1
            
            sys.stdout = open(f"./output/{self.case_id}_metrics_{i}.txt", 'w')
            getDataMetrics(self.case_id, cxs, cys, widths, heights)
            sys.stdout = terminal
        else:
            getDataMetrics(self.case_id, cxs, cys, widths, heights)
    
    def storeGraphs(self, ts, cxs, cys, widths, heights):
        if self.store_graphs:
            print("Creating Graphs for time-series data and their respective distributions")
            detectionGraphics(self.case_id, ts, cxs, cys, widths, heights, self.store_graphs)           
     
    def storeFit(self, ts, cys):
        if self.store_fit:
            params = getParams(f"Provide the true values of the dataset in float values: {self.case_id}  m, k, b:")
            defaultFitment(self.case_id, params, ts, cys, self.store_graphs) 
     
    def shapeMeasurements(self, cxs, cys):
        if self.model_name == "MultivariableSimpleHarmonicOscillator2D":
            zs = [[[a], [b]] for a, b in zip((mean(cxs) - cxs), (mean(cys) - cys))]
            us = zeros_like(zs)
        else:
            zs = [[[a]] for a in (mean(cys) - cys)]
            us = zeros_like(zs)
            
        return zs, us
    
    def storeEvaluation(self, ts, dts, zs, us):
        if self.store_eval:
            self.evaluation.defaultEvaluation(ts, dts, zs, us, store=self.store_graphs)
        else:
            pass
     
    def fullTest(self):    
        # If true, we're creating a new dataset from a video file, else we're reading an existing dataset from a file
        measurements = self.sourceMeasurements()
        for measurement in zip(*measurements):
            ts, dts, cxs, cys, widths, heights = measurement
            # If true, we store the metrics from the dataset in a .txt file, else we display our metrics to terminal only    
            self.storeMetrics(cxs, cys, widths, heights)
            
            # If true, create graphs for the timeseries and plot their distributions
            self.storeGraphs(ts, cxs, cys, widths, heights)

            # If true, we want to fit a synthetic function to the actual dataset, else pass
            self.storeFit(ts, cys)
            
            # Shape our measurements depending on the structure of our system model {A, B, H, Q, R}
            zs, us = self.shapeMeasurements(cxs, cys)
            
            # If true, we run the MMAE algorithm on our measurement data
            self.storeEvaluation(ts, dts, zs, us)
        
if __name__ == "__main__":
    # harness_config_path = "./configuration_files/harness_configs/sport/harness_sport_fourpass.json" 
    # harness_config_path = "./configuration_files/harness_configs/suv/harness_suv_threepass.json" 
    harness_config_path = "./configuration_files/harness_configs/truck/harness_truck_nopass.json" 
    harness = Harness(harness_config_path)
    harness.fullTest()
    plt.show()   
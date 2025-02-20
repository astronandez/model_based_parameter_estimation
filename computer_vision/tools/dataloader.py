import os
import csv
import time
from numpy import ndarray, loadtxt, array

class Dataloader:
    directory: str
    
    def __init__(self, directory: str):
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)
        
    def save(self, data: dict, header: list):
        for id, rows in data.items():
            csv_file = os.path.join(self.directory, f"obj_{id}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
            file_exists = os.path.isfile(csv_file)
            
            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(header)
                
                writer.writerows(rows)
        return csv_file
                
    def load(self, path: str):
        data = loadtxt(path, delimiter=',', skiprows=1)
        
        times = array([row[0] for row in data])
        dts = array([row[1] for row in data])
        cxs = array([row[2] for row in data])
        cys = array([row[3] for row in data])
        widths = array([row[4] for row in data])
        heights = array([row[5] for row in data])
        
        return times, dts, cxs, cys, widths, heights
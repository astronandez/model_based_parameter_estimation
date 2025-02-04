import os
import csv
import time
from numpy import ndarray, loadtxt, array

class Dataloader:
    directory: str
    
    def __init__(self, directory: str):
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)
        
    def save(self, data, header):
        for id, rows in data.items():
            csv_file = os.path.join(self.directory, f"obj_{id}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
            file_exists = os.path.isfile(csv_file)
            
            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(header)
                
                writer.writerows(rows)
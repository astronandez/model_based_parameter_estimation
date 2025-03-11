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
    
if __name__ == "__main__":
    from grapher import plotTimeSeries, plt
    from numpy import mean, float32, full
    import cv2 as cv
    import json
    
    
    case_id = "sport_nopass_rb_norm"
    data_file = f"./output/dslr data/{case_id}.csv"
    case_id2 = "sport_nopass_fb_norm"
    data_file2 = f"./output/dslr data/{case_id2}.csv"
    case_id3 = "sport_nopass_rw_norm"
    data_file3 = f"./output/dslr data/{case_id3}.csv"
    case_id4 = "sport_nopass_fw_norm"
    data_file4 = f"./output/dslr data/{case_id4}.csv"
    # case_id = "sport_nopass_rb"
    # data_file = f"./output/dslr data/{case_id}.csv"
    # case_id2 = "sport_nopass_fb"
    # data_file2 = f"./output/dslr data/{case_id2}.csv"
    # case_id3 = "sport_nopass_rw"
    # data_file3 = f"./output/dslr data/{case_id3}.csv"
    # case_id4 = "sport_nopass_fw"
    # data_file4 = f"./output/dslr data/{case_id4}.csv"
        
    # case_id = "sport_nopass_rb_norm"
    # data_file = f"./output/{case_id}.csv"
    # case_id2 = "sport_nopass_fb_norm"
    # data_file2 = f"./output/{case_id2}.csv"
    # case_id3 = "sport_nopass_rw_norm"
    # data_file3 = f"./output/{case_id3}.csv"
    # case_id4 = "sport_nopass_fw_norm"
    # data_file4 = f"./output/{case_id4}.csv"
    # case_id = "sport_nopass_rb"
    # data_file = f"./output/{case_id}.csv"
    # case_id2 = "sport_nopass_fb"
    # data_file2 = f"./output/{case_id2}.csv"
    # case_id3 = "sport_nopass_rw"
    # data_file3 = f"./output/{case_id3}.csv"
    # case_id4 = "sport_nopass_fw"
    # data_file4 = f"./output/{case_id4}.csv"
    dataloader = Dataloader("./output/")
    
    labels_diff = [f"./graphs/rb_rw_diff_sport_nopass_y_timeseries.fig",
            f"Timeseries data of y measurements Difference between body and wheel",
            "Time (s)",
            "Position (px)"]
    
    labels = [f"./graphs/{case_id}_nopass_y_timeseries.fig",
            f"Timeseries data of y measurements file: {case_id}",
            "Time (s)",
            "Position (px)"]
    
    labels2 = [f"./graphs/{case_id2}_nopass_y_timeseries.fig",
            f"Timeseries data of y measurements file: {case_id2}",
            "Time (s)",
            "Position (px)"]
    
    labels3 = [f"./graphs/{case_id3}_nopass_y_timeseries.fig",
            f"Timeseries data of y measurements file: {case_id3}",
            "Time (s)",
            "Position (px)"]
    
    labels4 = [f"./graphs/{case_id4}_nopass_y_timeseries.fig",
        f"Timeseries data of y measurements file: {case_id4}",
        "Time (s)",
        "Position (px)"]
    
    ts, _, cxs, cys, _, _ = dataloader.load(data_file)
    ts2, _, cxs2, cys2, _, _  = dataloader.load(data_file2)
    ts3, _, cxs3, cys3, _, _  = dataloader.load(data_file3)
    ts4, _, cxs4, cys4, _, _  = dataloader.load(data_file4)
    
    # cys = max(cys) - cys
    # cys2 = max(cys2) - cys2
    
    # min_length = min(len(cys2), len(cys))
    # y = cys[-min_length:] - cys2[-min_length:]
    # cys = max(cys) - cys
    # cys2 = max(cys2) - cys2
    # y = cys - cys2
 
    # plotTimeSeries((ts[-min_length:] - ts[0]), (y - mean(y)), labels_diff)
    # plotTimeSeries((ts[-min_length:] - ts[-min_length]), (cys[-min_length:] - mean(cys[-min_length:])), labels)
    # plotTimeSeries((ts2[-min_length:] - ts2[-min_length]), (cys2[-min_length:] - mean(cys2[-min_length:])), labels2)
    
    # plotTimeSeries((ts- ts[0]), y, labels_diff)
    
    cys = mean(cys) - cys
    cys2 = mean(cys2) - cys2
    cys3 = mean(cys3) - cys3
    cys4 = mean(cys4) - cys4
    
    plotTimeSeries((ts - ts[0]), cys, labels)
    plotTimeSeries((ts2 - ts2[0]), cys2, labels2)
    plotTimeSeries((ts3 - ts3[0]), cys3, labels3)
    plotTimeSeries((ts4 - ts4[0]), cys4, labels4)
    plt.show()
    
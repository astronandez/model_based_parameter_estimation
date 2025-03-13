import os
import csv
import time
from numpy import ndarray, column_stack, loadtxt, savetxt, array, arange, zeros_like, full_like

class Dataloader:
    directory: str
    
    def __init__(self, directory: str):
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)
        
    def save(self, data: dict, header: list):
        if not data:
            print("Warning: No data to save.")
            return None  # Return None when no data is provided

        csv_file = None  # Initialize csv_file to ensure it's always defined
        
        for id, rows in data.items():
            csv_file = os.path.join(self.directory, f"{id}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
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
        print(cys)
        return times, dts, cxs, cys, widths, heights
    
if __name__ == "__main__":
    from grapher import plotTimeSeries, plt
    from numpy import mean, float32, full
    import cv2 as cv
    import json
    
    dataloader = Dataloader("./output/")

    data_files = {
        "tag_3_n": "./data/apriltag/sport_nopass_apriltag_tag_3.csv",
        "tag_4_n": "./data/apriltag/sport_nopass_apriltag_tag_4.csv",
        "tag_5_n": "./data/apriltag/sport_nopass_apriltag_tag_5.csv",
        "tag_6_n": "./data/apriltag/sport_nopass_apriltag_tag_6.csv",
        "tag_3": "./data/apriltag/sport_onepass_apriltag_tag_3.csv",
        "tag_4": "./data/apriltag/sport_onepass_apriltag_tag_4.csv",
        "tag_5": "./data/apriltag/sport_onepass_apriltag_tag_5.csv",
        "tag_6": "./data/apriltag/sport_onepass_apriltag_tag_6.csv",
    }

    # Load data
    data = {tag: dataloader.load(path) for tag, path in data_files.items()}

    # Extract relevant data and compute mean-shifted values
    dt = 1 / 119.95
    x_series = {tag: (cxs - cxs[0]) for tag, (_, _, cxs, _, _, _) in data.items()}
    y_series = {tag: -(cys - cys[0]) for tag, (_, _, _, cys, _, _) in data.items()}
    time_series = {tag: arange(len(y)) * dt for tag, y in y_series.items()}

    # Define tag pairs and corresponding labels
    pairs = [
        ("tag_3", "tag_4"),
        ("tag_5", "tag_6"),
        ("tag_3_n", "tag_4_n"),
        ("tag_5_n", "tag_6_n"),
    ]

    y_difference_labels = {
        f"{a}_minus_{b}": [
            f"./graphs/{a}_minus_{b}_y_timeseries.fig",
            f"Difference: {a} - {b}",
            "Time (s)",
            "Position Difference (px)"
        ]
        for a, b in pairs
    }
    
    x_difference_labels = {
    f"{a}_minus_{b}_x": [
        f"./graphs/{a}_minus_{b}_x_timeseries.fig",
        f"X Difference: {a} - {b}",
        "Time (s)",
        "Position Difference (px)"
    ]
    for a, b in pairs
}

    for tag_a, tag_b in pairs:
        min_len = min(len(y_series[tag_a]), len(y_series[tag_b]))

        # Truncate series
        y_diff = y_series[tag_a][:min_len] - y_series[tag_b][:min_len]
        x_diff = x_series[tag_a][:min_len] - x_series[tag_b][:min_len]
        time_diff = time_series[tag_a][:min_len]

        # Save to CSV
        dts = full_like(y_diff, dt)
        output_data = column_stack((time_diff, dts, x_series[tag_b][:min_len], y_diff))
        output_path = f"./output/{tag_a}_minus_{tag_b}_corrected.csv"

        savetxt(output_path, output_data, 
                delimiter=",", 
                header="time, dts, Center (x-axis), Center (y-axis)", 
                comments='')

        print(f"Data saved successfully to {output_path}")

        plotTimeSeries(time_diff, y_diff, y_difference_labels[f"{tag_a}_minus_{tag_b}"])
        plotTimeSeries(time_diff, x_diff, x_difference_labels[f"{tag_a}_minus_{tag_b}_x"])

    plt.show()
    
    
    # # Save to CSV
    # for tag in y_series:
    #     dts = full_like(y_series[tag], dt)
    #     output_data = column_stack((time_series[tag], dts, x_series[tag], y_series[tag]))
    #     output_path = f"./output/sport_onepass_{tag}_corrected.csv"

    #     savetxt(output_path, output_data, 
    #             delimiter=",", 
    #             header="time, dts, Center (x-axis), Center (y-axis)", 
    #             comments='')
        
    #     print(f"Data saved successfully to {output_path}")
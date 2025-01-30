import csv
import json
from numpy import ndarray, loadtxt, array
from grapher import Grapher, plt

class Dataloader:
    def __init__(self):
        self.indicies = array([0])
        self.time = array([0])
        self.dt = array([0])
        self.cx = array([0])
        self.cy = array([0])
        self.width = array([0])
        self.height = array([0])
    
    def storeData(self, data, header, path):
        with open(path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(data)

    def loadMeasurements(self, path):
        data = loadtxt(path, delimiter=',', skiprows=1)
        
        self.indicies = array([row[0] for row in data])
        self.time = array([row[1] for row in data])
        self.dt = array([row[2] for row in data])
        self.cx = array([row[3] for row in data])
        self.cy = array([row[4] for row in data])
        self.width = array([row[5] for row in data])
        self.height = array([row[6] for row in data])
    
    def loadFittment(self, path):
        data = loadtxt(path, delimiter=',', skiprows=1)
        
        self.index_fit = array([row[0] for row in data])
        self.t = array([row[1] for row in data])
        self.z = array([row[2] for row in data])
    
    def loadSpringConstant(self, path):
        data = loadtxt(path, delimiter=',', skiprows=3, usecols=(0, 1))
        self.force = array([float(row[0]) for row in data])
        self.disp = array([float(row[1]/1000) for row in data])
        

if __name__ == "__main__":
    dataloader = Dataloader()
    grapher = Grapher()
    
    ################ Load and Normalize Tire Detection Measurements ##########################
    # model_id1 = 'lemur_sport1'
    # model_id2 = 'lemur_sport2'
    # model_id3 = 'lemur_sport3'
    # model_id4 = 'lemur_suv1'
    # model_id5 = 'lemur_suv2'
    # model_id6 = 'lemur_suv3'
    # model_id7 = 'lemur_sport_full1'
    # model_id8 = 'lemur_sport_full2'
    # model_id9 = 'lemur_sport_full3'
    # model_id10 = 'lemur_suv_twopass1'
    # model_id11 = 'lemur_suv_twopass2'
    # model_id12 = 'lemur_suv_twopass3'
    
    # measurementMetrics(f"./configurations/{model_id1}.json", f"./runs/{model_id1}/{model_id1}_back_tire")
    # measurementMetrics(f"./configurations/{model_id2}.json", f"./runs/{model_id2}/{model_id2}_back_tire")
    # measurementMetrics(f"./configurations/{model_id3}.json", f"./runs/{model_id3}/{model_id3}_back_tire")
    # measurementMetrics(f"./configurations/{model_id4}.json", f"./runs/{model_id4}/{model_id4}_back_tire")
    # measurementMetrics(f"./configurations/{model_id5}.json", f"./runs/{model_id5}/{model_id5}_back_tire")
    # measurementMetrics(f"./configurations/{model_id6}.json", f"./runs/{model_id6}/{model_id6}_back_tire")
    # measurementMetrics(f"./configurations/{model_id7}.json", f"./runs/{model_id7}/{model_id7}_back_tire")
    # measurementMetrics(f"./configurations/{model_id8}.json", f"./runs/{model_id8}/{model_id8}_back_tire")
    # measurementMetrics(f"./configurations/{model_id9}.json", f"./runs/{model_id9}/{model_id9}_back_tire")
    # measurementMetrics(f"./configurations/{model_id10}.json", f"./runs/{model_id10}/{model_id10}_back_tire")
    # measurementMetrics(f"./configurations/{model_id11}.json", f"./runs/{model_id11}/{model_id11}_back_tire")
    # measurementMetrics(f"./configurations/{model_id12}.json", f"./runs/{model_id12}/{model_id12}_back_tire")
    
    # measurementMetrics(f"./configurations/{model_id12}.json", f"./runs/{model_id12}/{model_id12}_rot_front_tire")
    # measurementMetrics(f"./configurations/{model_id12}.json", f"./runs/{model_id12}/{model_id12}_rot_back_tire")
    
    # dataloader.loadSpringConstant("./runs/spring_constant_tests/spring3.csv")
    # grapher.plotSpringConstant('./runs/spring_constant_tests/spring3', dataloader.disp, dataloader.force)
    # plt.show()
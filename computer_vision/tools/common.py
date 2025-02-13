from numpy import arange, array, eye, zeros, mean, var, std, sqrt, square, zeros_like, argmin, asarray
import json
import cv2 as cv
from itertools import product
from ..tools.dataloader import Dataloader
from .grapher import *

############## Config Utilities #################
def loadConfig(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def defaultSetup(config):
    Q_start = config["Q_variants_start"]
    Q_end = config["Q_variants_end"]
    Q_step = config["Q_variants_step"]

    R_start = config["R_variants_start"]
    R_end = config["R_variants_end"]
    R_step = config["R_variants_step"]
    
    m_start = config["m_variants_start"]
    m_end = config["m_variants_end"]
    m_step = config["m_variants_step"]

    k_start = config["k_variants_start"]
    k_end = config["k_variants_end"]
    k_step = config["k_variants_step"]

    b_start = config["b_variants_start"]
    b_end = config["b_variants_end"]
    b_step = config["b_variants_step"]

    # Generate model variants
    ms = arange(m_start, m_end + m_step, m_step).tolist()
    ks = arange(k_start, k_end + k_step, k_step).tolist()
    bs = arange(b_start, b_end + b_step, b_step).tolist()

    Qs = arange(Q_start, Q_end + Q_step, Q_step).tolist()
    Rs = arange(R_start, R_end + R_step, R_step).tolist()

    # Generate all possible combinations of m, k, and b
    λs = [array(λ) for λ in product(ms, ks, bs)]
    # Configure Matricies
    x0 = array(config["x0"])
    dt = config["dt"]
    m = config['true_m']
    k = config['true_k']
    b = config['true_b']
    H = array(config["H"])
    Q = eye(H.shape[1]) * config["true_Q"]
    R = eye(H.shape[0]) * config["true_R"]
    
    return m, k, b, Q, R, λs, dt, H, Qs, Rs, x0

############# Data Utilities ###################
def defaultMetrics(data):
    data_mean = mean(data)
    data_var = var(data)
    data_std = std(data)
    return data_mean, data_var, data_std
    
def inspectData(model_id, ts, cxs, cys, widths, heights, store=True):
    labels1 = [f"./graphs/{model_id}_y_measurements.fig",
                f"Detector Measurements, position in frame (y-axis)",
                "Time (s)",
                "Position (px)"]
    labels2 = [f"./graphs/{model_id}_x_measurements.fig",
                f"Detector Measurements, position in frame (x-axis)",
                "Time (s)",
                "Position (px)"]
    labels3 = [f"./graphs/{model_id}_width_measurements.fig",
                f"Detector Measurements, width of object over time",
                "Time (s)",
                "Magnitude (px)"]
    labels4 = [f"./graphs/{model_id}_height_measurements.fig",
                f"Detector Measurements, height of object over time",
                "Time (s)",
                "Magnitude (px)"]
    
    label_cys = [f"./graphs/{model_id}_cys_distribution.fig",
                f'Probability Distribution {model_id} y-axis center (Mean = 0)',
                "Position (px)",
                "Probability Density"]
    label_cxs = [f"./graphs/{model_id}_cxs_distribution.fig",
                f'Probability Distribution {model_id} x-axis center (Mean = 0)',
                "Position (px)",
                "Probability Density"]
    label_height = [f"./graphs/{model_id}_height_distribution.fig",
                f'Probability Distribution {model_id} object height (Mean = 0)',
                "Magnitude (px)",
                "Probability Density"]
    label_width = [f"./graphs/{model_id}_width_distribution.fig",
                f'Probability Distribution {model_id} object width (Mean = 0)',
                "Magnitude (px)",
                "Probability Density"]
    
    plotTimeSeries(ts, cys, labels1, store)
    plotTimeSeries(ts, cxs, labels2, store)
    plotTimeSeries(ts, widths, labels3, store)
    plotTimeSeries(ts, heights, labels4, store)
    plotDistribution(cys, label_cys, store)
    plotDistribution(cxs, label_cxs, store)
    plotDistribution(widths, label_width, store)
    plotDistribution(heights, label_height, store)

############## Input Utilities #################
def impulse(steps, step_impulse, amplitude, dt):
    rescale_amp = amplitude / dt
    u = [array([0.0]) for i in range(steps)]
    u[step_impulse][0] = amplitude
    return array(u)

def ramp(t, impulse_time, amplitude):
    u = zeros_like(t)

    for i in range(impulse_time + 1):
        u[i][0] = (i / impulse_time) * amplitude
        
    for i in range(impulse_time, 2 * impulse_time):
        u[i] = ((2 * impulse_time - i) / (2 * impulse_time - impulse_time)) * amplitude
    
    return u
    
def squareWave(start, length, width, amplitude):
    signal = zeros(length)
    end = min(length, start + width)  # Ensure the end doesn't exceed the signal length
    signal[start:end] = amplitude
    u = [array([[value]]) for value in signal]
    return u

def step_function(steps, amplitude, change=100):
    """
    Generate a step function input signal.
    
    :param steps: The total number of time steps.
    :param amplitude: The amplitude of the step.
    :return: A numpy array representing the step function input signal.
    """
    signal = zeros(steps)
    signal[change:] = amplitude
    u = [array([[value]]) for value in signal]
    return u

############## CV Utilities ####################
def centerToBoundingBox(cx, cy, w, h):

    x1 = cx - (0.5 * w)
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    
    return x1, y1, x2, y2

def classToWeight(car_type_label):
    classes = ['Convertible', 'Crossover', 'Fastback', 'Hardtop Convertible', 'Hatchback', 'MPV', 'Minibus', 'Pickup Truck', 'SUV', 'Sedan', 'Sports', 'Wagon']

    if car_type_label == 0:
        return 4462
    elif car_type_label == 1:
        return 4331
    elif car_type_label == 2:
        return 3743
    elif car_type_label == 3:
        return 2992
    elif car_type_label == 4:
        return 4451
    elif car_type_label == 5:
        return 3448
    elif car_type_label == 6:
        return 3695
    elif car_type_label == 7:
        return 4433
    elif car_type_label == 8:
        return 3466
    elif car_type_label == 9:
        return 3494
    elif car_type_label == 10:
        return 4037
    elif car_type_label == 11:
        return 3092
    else:
        return 0

def drawDetections(frame, cx, cy, width, height):
    x1, y1, x2, y2 = centerToBoundingBox(cx, cy, width, height)
    cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv.putText(frame, f'z: {cy} px', (int(cx), int(cy) - int(height/2 + 5)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame



if __name__ == "__main__":
    pass
    # dataloader = Dataloader()
    # dataloader2 = Dataloader()
    # config_path = "./configurations/lemur_sticky_sport2.json"
    # config = loadConfig(config_path)
    # dataloader.loadMeasurements(f"{config['measurements_output'][:-4]}_lower{config['measurements_output'][-4:]}")
    # dataloader2.loadMeasurements(f"{config['measurements_output'][:-4]}_upper{config['measurements_output'][-4:]}")

    # start = config['t_start']
    # end = config['t_end']
    
    # t1, z1 = dataloader.time, dataloader.cy
    # t2, z2 = dataloader2.time, dataloader2.cy
    # r1_seg = z1[start:end]
    # r2_seg = z2[start:end]
    
    # measurements = []
    # for i in range(len(t1)):
    #     z = (z2[i] - sqrt(mean(square(r2_seg)))) - (z1[i] - sqrt(mean(square(r1_seg))))
    #     measurements.append(z)
    
    # print("Measurement mean(z) = ", mean(r1_seg))
    # print(f"Measurement Varience var(z) = {var(r1_seg)}")
    # print(f"Measurement Standard Deviation std(z) = {std(r1_seg)}")
    # print(f"Initial Position = {r1_seg[0]}")
    # print("Measurement mean(z) = ", mean(r2_seg))
    # print(f"Measurement Varience var(z) = {var(r2_seg)}")
    # print(f"Measurement Standard Deviation std(z) = {std(r2_seg)}")
    # print(f"Initial Position = {r2_seg[0]}")
    
    # # grapher.plotMeasurements(f"{config_path}_lower", f"{config['graph_output']}_lower", dataloader.time[start:end], (dataloader.cy[start:end] - sqrt(mean(square(dataloader.cy[start:end])))))
    # # grapher.plotMeasurements(f"{config_path}_upper", f"{config['graph_output']}_upper", dataloader2.time[start:end], (dataloader2.cy[start:end] - sqrt(mean(square(dataloader2.cy[start:end])))))
    
    # # grapher.plotMeasurements(f"{config_path}_lower", f"{config['graph_output']}_lower", t1, (z1 - sqrt(mean(square(r1_seg)))))
    # # grapher.plotMeasurements(f"{config_path}_upper", f"{config['graph_output']}_upper", t2, (z2 - sqrt(mean(square(r2_seg)))))
    # # grapher.plotMeasurements(config_path, config['graph_output'], t2, measurements)
    
    # full_data = zip(list(range(0, len(t1))), t1, dataloader2.dt, dataloader2.cx, measurements, dataloader2.width, dataloader2.height)
    # header = ["index", "time", "dt", "Center (x-axis)", "Center (y-axis)", "box width", "box height"]
    # dataloader.storeData(full_data, header, f"{config['measurements_output']}")
    # plt.show()
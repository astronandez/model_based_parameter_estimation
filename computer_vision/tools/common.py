from numpy import arange, array, eye, zeros, mean, var, std, sqrt, square, zeros_like, argmin, asarray
import json
import cv2 as cv
from itertools import product
from .grapher import *

############## Config Utilities #################
def loadConfig(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def defaultSetup(config):
    model_name = config["model_name"]
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
    
    return m, k, b, Q, R, λs, dt, H, Qs, Rs, x0, model_name

############# Data Utilities ###################
def defaultMetrics(data):
    data_mean = mean(data)
    data_var = var(data)
    data_std = std(data)
    return data_mean, data_var, data_std
    
def detectionGraphics(case_id, ts, cxs, cys, widths, heights, store=True):
    labels1 = [f"./graphs/{case_id}_y_measurements.fig",
                f"Detector Measurements {case_id}, position in frame (y-axis)",
                "Time (s)",
                "Position (px)"]
    labels2 = [f"./graphs/{case_id}_x_measurements.fig",
                f"Detector Measurements {case_id}, position in frame (x-axis)",
                "Time (s)",
                "Position (px)"]
    labels3 = [f"./graphs/{case_id}_width_measurements.fig",
                f"Detector Measurements {case_id}, width of object over time",
                "Time (s)",
                "Magnitude (px)"]
    labels4 = [f"./graphs/{case_id}_height_measurements.fig",
                f"Detector Measurements {case_id}, height of object over time",
                "Time (s)",
                "Magnitude (px)"]
    
    label_cys = [f"./graphs/{case_id}_y_distribution.fig",
                f'Probability Distribution {case_id} y-axis center (Mean = 0)',
                "Position (px)",
                "Probability Density"]
    label_cxs = [f"./graphs/{case_id}_x_distribution.fig",
                f'Probability Distribution {case_id} x-axis center (Mean = 0)',
                "Position (px)",
                "Probability Density"]
    label_height = [f"./graphs/{case_id}_height_distribution.fig",
                f'Probability Distribution {case_id} object height (Mean = 0)',
                "Magnitude (px)",
                "Probability Density"]
    label_width = [f"./graphs/{case_id}_width_distribution.fig",
                f'Probability Distribution {case_id} object width (Mean = 0)',
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

def getDataMetrics(case_id, cxs, cys, widths, heights):
    cy_mean, cy_var, cy_std = defaultMetrics(cys)
    cx_mean, cx_var, cx_std = defaultMetrics(cxs)
    wid_mean, wid_var, wid_std = defaultMetrics(widths)
    h_mean, h_var, h_std = defaultMetrics(heights)
        
    print(f"Model Metrics: {case_id}")
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
    print("Standard Deviation of detection height:", h_std, "\n") 

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

def drawDetections(frame, id, cx, cy, width, height):
    x1, y1, x2, y2 = centerToBoundingBox(cx, cy, width, height)
    cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv.putText(frame, f'z: {cy} px', (int(cx), int(cy) - int(height/2 + 5)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv.putText(frame, f'id: {id}', (int(cx), int(cy) + int(height/2 + 15)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

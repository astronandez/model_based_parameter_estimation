import sys
from numpy import arange, newaxis
from numpy.random import seed
from parameter_estimation_pipeline.System.system_simulator import SystemSimulator
from parameter_estimation_pipeline.System.Model.system import System
from parameter_estimation_pipeline.MMAE.mmae import MMAE
from components.tools.common import *
from components.tools.grapher import plotValidateSyntheticSystems         
                
def loop(simulators: list[SystemSimulator], t, u, dt, mmaes: list[MMAE] = []):
    sim_xs = []
    sim_zs = []
    sim_vs = []
    sim_ws = []
    sim_λ_hats = []
    sim_posteriors = []
    sim_pdvs = []

    for i in range(len(simulators)):
        cur_xs = []
        cur_zs = []
        cur_vs = []
        cur_ws = []
        cur_post = []
        cur_λhat = []
        cur_pdvs = []
        
        for j in range(len(t)):
            x, z = simulators[i].update([u[j]], dt)
            
            if mmaes:
                λ_hat, cumulative_posteriors, pdvs = mmaes[i].update(u[j], z, dt)
                cur_post.append(cumulative_posteriors)
                cur_pdvs.append(pdvs)
                cur_λhat.append(λ_hat)
                
            cur_vs.append(simulators[i].plant.h.v[-1].flatten())
            cur_ws.append(simulators[i].plant.h.w[-1].flatten())
            cur_xs.append(x.flatten())
            cur_zs.append(z.flatten()) 
            
        if mmaes:
            sim_posteriors.append(cur_post)
            sim_pdvs.append(cur_pdvs)
            sim_λ_hats.append(cur_λhat)
        
        sim_xs.append(cur_xs)
        sim_zs.append(cur_zs)
        sim_vs.append(cur_vs)
        sim_ws.append(cur_ws)
        
    return sim_xs, sim_zs, sim_vs, sim_ws, sim_λ_hats, sim_posteriors, sim_pdvs

def makeSyntheticSystems(config):
    seed(42)
    m, k, b, Q, R, λs, dt, H, Qvar, Rvar, x0 = defaultSetup(config)
    systems: list[SystemSimulator] = []
    models: list[System] = []
    for λ in λs:
        for i in Qvar:
            for j in Rvar:
                Qk = eye(H.shape[1]) * i
                Rk = eye(H.shape[0]) * j
                s = SystemSimulator(λ, dt, H, Qk, Rk, x0, True)
                systems.append(s)
                models.append(s.model)
    
    return systems, models

def validateSyntheticSystems(config, output_path, t, u, dt, reso, labels):
    systems, models = makeSyntheticSystems(config)
    sim_xs, sim_zs, sim_vs, sim_ws, _, _, _ = loop(systems, t, u, dt)
    plotValidateSyntheticSystems(models, sim_zs, t, u, reso, labels)
    
    sys.stdout = open(output_path, 'w')
    for i in range(len(sim_vs)):
        mean_v = mean(sim_vs[i])
        var_v = var(sim_vs[i])
        stdd_v = std(sim_vs[i])
        
        mean_w = mean(sim_ws[i])
        var_w = var(sim_ws[i])
        stdd_w = std(sim_ws[i])
        
        print(f"Model: m:{systems[i].model.m}, k:{systems[i].model.k}, b:{systems[i].model.b}, Q:{systems[i].model.Q[0,0]}, R:{systems[i].model.R[0,0]}")
        print("===== Measurement Noise Metrics =====")
        print("Mean of v:", mean_v) 
        print("Variance of v:", var_v)
        print("Standard Deviation of v:", stdd_v)
        print("======== System Noise Metrics =======")
        print("Mean of w:", mean_w) 
        print("Variance of w:", var_w)
        print("Standard Deviation of w:", stdd_w, "\n")   
    sys.stdout.close()
    
if __name__ == "__main__":
    config_level0_path = "./configuration_files/validation_configs/level0.json"
    config_level1_path = "./configuration_files/validation_configs/level1.json"
    config_level2_path = "./configuration_files/validation_configs/level2.json"
    
    config_level0 = loadConfig(config_level0_path)
    config_level1 = loadConfig(config_level1_path)
    config_level2 = loadConfig(config_level2_path)
            
    level0_output = "./output/level0.txt"
    level1_output = "./output/level1.txt"
    level2_output = "./output/level2.txt"
    
    start, stop, dt, amplitude, reso = 0.0, 10.0, 0.05, 2000, 1
    t = arange(start, stop + dt, dt)[:, newaxis]
    # u = impulse(len(t), 0, amplitude, dt)
    u = ramp(t, 1, amplitude)
    # u = step_function(len(t), amplitude, change=0)
    
    labels0 = ["./graphs/level0_dt_0_05.fig",
             f"Validation Graph of {config_level0}: R = 0, Q = 0",
              "Time (s)",
              "Displacement (m)",
              "Force(N)"]
    
    labels1 = ["./graphs/level1_dt_0_05.fig",
             f"Validation Graph of {config_level1}: Varying R, Q = 0",
              "Time (s)",
              "Displacement (m)",
              "Force(N)"]
    
    labels2 = ["./graphs/level2_dt_0_05.fig",
             f"Validation Graph of {config_level2}: R = 0 , Varying Q",
              "Time (s)",
              "Displacement (m)",
              "Force(N)"]
    
    validateSyntheticSystems(config_level0, level0_output, t, u, dt, reso, labels0)
    validateSyntheticSystems(config_level1, level1_output, t, u, dt, reso, labels1)
    validateSyntheticSystems(config_level2, level2_output, t, u, dt, reso, labels2)
    plt.show()
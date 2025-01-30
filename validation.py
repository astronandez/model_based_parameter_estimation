from numpy import ndarray, linspace, arange, array,zeros
from numpy.random import seed
from parameter_estimation_pipeline.System.system_simulator import SystemSimulator
from parameter_estimation_pipeline.MMAE.mmae import MMAE
from evaluation import Evaluation
from components.tools.grapher import Grapher
from components.tools.common import *

def loopSimulators(simulators: list[SystemSimulator], mmaes: list[MMAE], t, u, dt):
    compiled_simulation_xs = []
    compiled_simulation_zs = []
    compiled_simulation_vs = []
    compiled_simulation_λ_hats = []
    compiled_simulation_posteriors = []
    compiled_pdvs_summary = []
    
    for i in range(len(simulators)):
        sim_xs = []
        sim_zs = []
        sim_vs = []
        sim_post = []
        sim_λhat = []
        sim_pdvs = []

        for k in range(len(t)):
            x, z = simulators[i].update(u[k], dt)
            λ_hat, cumulative_posteriors, pdvs = mmaes[i].update(u[k], z, dt)

            sim_post.append(cumulative_posteriors)
            sim_pdvs.append(pdvs)
            sim_λhat.append(λ_hat)
            sim_vs.append(simulators[i].plant.h.v[-1])
            sim_xs.append(x)
            sim_zs.append(z.flatten())
        
        compiled_simulation_posteriors.append(sim_post)
        compiled_pdvs_summary.append(sim_pdvs)
        compiled_simulation_λ_hats.append(sim_λhat)
        compiled_simulation_xs.append(sim_xs)
        compiled_simulation_zs.append(sim_zs)
        compiled_simulation_vs.append(sim_vs)

        
    return compiled_simulation_xs, compiled_simulation_zs, compiled_simulation_vs, compiled_simulation_λ_hats, compiled_pdvs_summary, compiled_simulation_posteriors

def validationMetrics(zs):
    z_vars = []
    z_stds = []
    z_mus = []
    
    for i in range(len(zs)):
        print(zs[i])
        z_sim = array(zs[i])
        z_mu = mean(z_sim)
        z = z_mu - z_sim
        z_var = var(z)
        z_std = std(z)
        
        z_vars.append(z_var)
        z_stds.append(z_std)
        z_mus.append(z_mu)
        print(f"Simulation {i}")
        print("Measurement mean(z) = ", z_mu)
        print(f"Measurement Varience(z) = {z_var}")
        print(f"Measurement Standard Deviation(z)  = {z_std}")
        
    # print("Measurement mean(z) = ", mean(z_mus))
    # print(f"Measurement Varience(z) = {mean(z_vars)}")
    # print(f"Measurement Standard Deviation(z)  = {mean(z_stds)}")
        
    return z_vars, z_stds, z_mus
    
def validationSyntheticSystems(config_path, t, u, graph_path):
    seed(42)
    grapher = Grapher()
    config, m, k, b, true_Q, true_R, λs, dt, H, Qs, Rs, x0 = level1Setup(config_path)
    mmaes: list[MMAE] = []
    systems: list[SystemSimulator] = []
    models = []
    true_λs = []

    for i in λs:
        for j in Qs:
            for k in Rs:
                Q = eye(H.shape[1]) * j
                R = eye(H.shape[0]) * k
                s = SystemSimulator(i, dt, H, true_Q, true_R, x0, True)
                mmae = MMAE(λs, dt, H, Q, R, x0, False)

                true_λs.append(array([m,k,b]))
                mmaes.append(mmae)
                systems.append(s)
                models.append(s.model)

    compiled_simulation_xs, compiled_simulation_zs, compiled_simulation_vs, compiled_simulation_λ_hats, compiled_pdvs_summary, compiled_simulation_posteriors = loopSimulators(systems, mmaes, t, u, config["dt"])
    grapher.plotGeneratedSystems(graph_path, config_path, t, u, compiled_simulation_zs, models, config['plot_every'])
    for i in range(len(compiled_simulation_λ_hats)):
        grapher.plot_λ_hat(config_path, graph_path, t, compiled_simulation_λ_hats[i], array([m, k, b]))
        # grapher.plot_heatmap(f"{graph_path}_Likelihood", pdvs_summary, times, evaluation.model_variants, title=f"{config_path}: Heatmap of Model Likelihood Over Time")
        # grapher.plot_heatmap(f"{graph_path}_Posteriors", cumulative_posteriors_summary, times, evaluation.model_variants, title=f"{config_path}: Heatmap of Cumulative Posterior Probabilities Over Time")
    z_vars, z_stds, z_mus = validationMetrics(compiled_simulation_vs)

def validationRealSystem(config_path, graph_path, start = None, end = None):
    grapher = Grapher()
    dataloader = Dataloader()
    evaluations: list[Evaluation] = []
    models = []
    
    config, λs, dt, H, Qs, Rs, x0 = level1Setup(config_path)
    dataloader.loadMeasurements(config['measurements_output'])
    t = dataloader.time[start:end] - dataloader.time[start]
    dts = dataloader.dt[start:end]
    z = dataloader.cy[start:end] - mean(dataloader.cy[start:end])
    u = [array([[0]]) for _ in range(len(t))]
    
    for i in λs:
        for j in Qs:
            for k in Rs:
                Q = eye(H.shape[1]) * j
                R = eye(H.shape[0]) * k
                evaluation = Evaluation(config, λs, dt, H, Q, R, x0, False)
                evaluations.append(evaluation)
                for m in evaluation.mmae.EstimatorLikelihoods:
                    models.append(m.Estimator.SystemSimulator.model)

    
    compiled_evaluation_zs = []
    for i in range(len(evaluations)):
        times, observed_z, estimator_ẑs, lambda_hats, pdvs_summary, cumulative_posteriors_summary = evaluations[i].run(dts, z)
        compiled_evaluation_zs.append(observed_z)
    
    grapher.plotGeneratedSystems(graph_path, config_path, t, u, compiled_evaluation_zs, models, 1, z)
    z_vars, z_stds, z_mus = validationMetrics(compiled_evaluation_zs)
                   
if __name__ == "__main__":
    # config_path1 = './configurations/lemur_sport1.json'
    # # config_path2 = './configurations/lemur_sport2.json'
    # # config_path3 = './configurations/lemur_sport3.json'
    # config1 = loadConfig(config_path1)
    # # config2 = loadConfig(config_path2)
    # # config3 = loadConfig(config_path3)
    # validationRealSystem(config_path1, config1["graph_output"], 60, 95)
    # # # validationRealSystem(config_path2, config2["graph_output"], 60, 95)
    # # # validationRealSystem(config_path3, config3["graph_output"], 60, 95)
    # plt.show()
    
    config_level0 = "./configurations/validation/level0.json"
    config_level1 = "./configurations/validation/level1.json"
    config_level2 = "./configurations/validation/level2.json"
    config_level4 = "./configurations/validation/level4.json"
    
    start, stop, dt, amplitude = 0.0, 10.0, 0.1, 1000
    t = arange(start, stop + dt, dt)
    # u = impulse(len(t), 10, amplitude)
    u = step_function(len(t), amplitude, change=int(1.0/dt))
    
    validationSyntheticSystems(config_level4, t, u, "./runs/validation_tests/level4")
    # validationSyntheticSystems(config_level1, t, u, "./runs/validation_tests/level1")
    # validationSyntheticSystems(config_level2, t, u, "./runs/validation_tests/level2")
    # validationSyntheticSystems()
    plt.show()
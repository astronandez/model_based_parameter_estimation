import json
import csv
from numpy import array, mean
from parameter_estimation_pipeline.MMAE.mmae import MMAE
from components.tools.common import *

class Evaluation:
    mmae: MMAE
    true_parameter: float
    model_variants: list[float]
    
    def __init__ (self, config, λs, dt, H, Q, R, x0, noisy):
        self.config = config
        mmae = MMAE(λs, dt, H, Q, R, x0, noisy)
        self.mmae = mmae
        self.model_variants = λs
        
    def nextMeasurement(self, dt, z):
        u = array([[0.0]])
        z = array([[z]])
        λ_hat, cumulative_posteriors, pdvs = self.mmae.update(u, z, dt)
        return λ_hat, cumulative_posteriors, pdvs
    
    def run(self, dts, zs):
        time_track = 0.0
        λ_final = 0
        observed_z = []
        estimator_ẑs = []
        times = []
        cumulative_posteriors_summary = []
        lambda_hats = []
        pdvs_summary = []
        
        for dt, z in zip(dts, zs):
            λ_hat, cumulative_posteriors, pdvs = self.nextMeasurement(dt, z)
            λ_final = λ_hat
            print(λ_hat)
            
            cumulative_posteriors_summary.append(cumulative_posteriors)
            pdvs_summary.append(pdvs)
            lambda_hats.append(λ_hat)
            time_track += dt
            times.append(time_track)
            observed_z.append(z)
        
        cumulative_posteriors_summary = array(cumulative_posteriors_summary)
        pdvs_summary = array(pdvs_summary)
        observed_z = array(observed_z)
        estimator_ẑs = array(estimator_ẑs)
        
        print(f'Final Parameter Estimates: {λ_final}')
        return times, observed_z, estimator_ẑs, lambda_hats, pdvs_summary, cumulative_posteriors_summary    

if __name__ == "__main__":
    from components.tools.grapher import Grapher, plt
    from components.tools.dataloader import Dataloader 
    
    def evaluationTestbench(config, measurements_path, λs, m, k, b, dt, H, Q, R, x0, graph_path, start = None, end = None):
        dataloader = Dataloader()
        grapher = Grapher()
        evaluation = Evaluation(config, λs, dt, H, Q, R, x0, False)
        
        dataloader.loadMeasurements(measurements_path)
        t = dataloader.time[start:end]
        dts = dataloader.dt[start:end]
        z = dataloader.cy[start:end]
        
        times, observed_z, estimator_ẑs, lambda_hats, pdvs_summary, cumulative_posteriors_summary = evaluation.run(dts, z)
        true_λ = [m, k, b]
        grapher.plotMeasurements(config_path, graph_path, t, z)
        grapher.plot_λ_hat(config_path, graph_path, times, lambda_hats, true_λ)
        grapher.plot_heatmap(f"{graph_path}_Likelihood", pdvs_summary, times, evaluation.model_variants, title=f"{config_path}: Heatmap of Model Likelihood Over Time")
        grapher.plot_heatmap(f"{graph_path}_Posteriors", cumulative_posteriors_summary, times, evaluation.model_variants, title=f"{config_path}: Heatmap of Cumulative Posterior Probabilities Over Time")
    
    grapher = Grapher()
    config_path = './configurations/lemur_sticky_sport_load.json'
    config, λs, m, k, b, dt, H, Q, R, x0 = evaluationSetup(config_path)
    # measurements_path = f"{config['measurements_output'][:-4]}_upper{config['measurements_output'][-4:]}"
    measurements_path = config['measurements_output']
    evaluationTestbench(config_path, measurements_path, λs, m, k, b, dt, H, Q, R, x0, config["graph_output"], 0 ,-1)
    plt.show()
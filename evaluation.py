import sys
from numpy import array, mean, ndarray, empty
from parameter_estimation_pipeline.MMAE.mmae import MMAE
from generator import defaultMeasurementGeneration
from computer_vision.tools.common import *

class Evaluation:
    mmae: MMAE
    model_variants: list[ndarray]

    def __init__(self, evaluation_config):
        m, k, b, Q, R, λs, dt, H, Qs, Rs, x0 = defaultSetup(evaluation_config)
        self.mmae = MMAE(λs, dt, H, Q, R, x0, False)
        self.model_variants = λs
    
    def update(self, dt, z):
        # u = array([[0.0], [0.0]])
        u = array([[0.0]])
        z = array([[z]])
        λ_hat, cumulative_posteriors, pdvs = self.mmae.update(u, z, dt)
        print(λ_hat)
        return λ_hat, cumulative_posteriors, pdvs
    
    def run(self, dts, zs):
        run_post = []
        run_λ_hat = []
        run_pdvs = []

        for dt, z in zip(dts, zs):
            # print(z)
            λ_hat, cumulative_posteriors, pdvs = self.update(dt, z)
            run_post.append(cumulative_posteriors)
            run_λ_hat.append(λ_hat)
            run_pdvs.append(pdvs)
            
        return array(run_λ_hat), array(run_post), array(run_pdvs)

if __name__ == "__main__":
    
    def defaultEvaluation(evaluation_config):
        dataloader = Dataloader("./output/")
        ts, dts, cxs, cys, widths, heights = dataloader.load(f"./data/{evaluation_config['model_id']}.csv")

        evaluation = Evaluation(evaluation_config)
        
        # Labels necessary for graph file path, and title
        label_lambda = [f"./graphs/{evaluation_config['model_id']}_estimations.fig",
                        f'{evaluation_config['model_id']}: Parameter Estimates (m, k, b) vs Time']
        label_likely = [f"./graphs/{evaluation_config['model_id']}_likelyhoods.fig",
                        f'{evaluation_config['model_id']}: Heatmap of Model Likelihood Over Time']
        label_poster = [f"./graphs/{evaluation_config['model_id']}_posteriors.fig",
                        f'{evaluation_config['model_id']}: Heatmap of Cumulative Posterior Probabilities Over Time']
        
        run_λ_hat, run_post, run_pdvs = evaluation.run(dts, (cys - mean(cys)))
        plotLambdaHat(ts, run_λ_hat, [evaluation_config["true_m"], evaluation_config["true_k"],evaluation_config["true_b"]], label_lambda)
        plotHeatmap(run_pdvs, ts, evaluation.model_variants, label_likely)
        plotHeatmap(run_post, ts, evaluation.model_variants, label_poster)
        plt.show()
     
    def default2DEvaluation(camera_config, detector_config, evaluation_config):
        ts, dts, cxs, cys = defaultMeasurementGeneration(camera_config, detector_config)
    
        exp_name = "m105_5_k80_80"
        inspectData(exp_name, ts, dts, cxs, cys)
        sys.stdout = open(f"./output/{exp_name}.txt", 'w')
        mean_y = mean(cys)
        var_y = var(cys - mean_y)
        stdd_y = std(cys - mean_y)
        
        mean_x = mean(cxs)
        var_x = var(cxs - mean_x)
        stdd_x = std(cxs - mean_x)
        
        print("===== Measurement Noise Metrics =====")
        print("Mean of y position:", mean_y) 
        print("Variance of y position:", var_y)
        print("Standard Deviation of y position:", stdd_y)
        print("======== System Noise Metrics =======")
        print("Mean of x position:", mean_x) 
        print("Variance of x position:", var_x)
        print("Standard Deviation of x position:", stdd_x, "\n")
        
        dataloader = Dataloader("./output/")
        ts, dts, cxs, cys, widths, heights = dataloader.load(f"./data/{exp_name}.csv")
        
        zs = empty((2, 2), dtype=object)

        # Assign cys to position (0,0) and cxs to (1,0)
        zs[0, 0] = cxs
        zs[1, 0] = cys
        
        evaluation = Evaluation(evaluation_config)
        label_lambda = [f"./graphs/{exp_name}_estimations.fig",
                        f'{exp_name}: Parameter Estimates (m, k, b) vs Time']
        label_likely = [f"./graphs/{exp_name}_likelyhoods.fig",
                        f'{exp_name}: Heatmap of Model Likelihood Over Time']
        label_poster = [f"./graphs/{exp_name}_posteriors.fig",
                        f'{exp_name}: Heatmap of Cumulative Posterior Probabilities Over Time']
        
        run_λ_hat, run_post, run_pdvs = evaluation.run(dts, zs)
        plotLambdaHat(ts, run_λ_hat, [evaluation_config["true_m"], evaluation_config["true_k"],evaluation_config["true_b"]], label_lambda)
        plotHeatmap(run_pdvs, ts, evaluation.model_variants, label_likely)
        plotHeatmap(run_post, ts, evaluation.model_variants, label_poster)
        plt.show()
        
    evaluation_config_path = "./configuration_files/evaluation_config.json"
    # camera_config_path = "./configuration_files/camera_configs/camera_generator.json"
    # detector_config_path = "./configuration_files/detector_configs/detector_base.json"
    
    evaluation_config = loadConfig(evaluation_config_path)
    # camera_config = loadConfig(camera_config_path)
    # detector_config = loadConfig(detector_config_path)
    
    defaultEvaluation(evaluation_config)
    # default2DEvaluation(camera_config, detector_config, evaluation_config)
import sys
from numpy import array, mean, ndarray, empty, diag, zeros_like
from parameter_estimation_pipeline.MMAE.mmae import MMAE
from computer_vision.tools.common import *

class Evaluation:
    mmae: MMAE
    model_variants: list[ndarray]

    def __init__(self, evaluation_config, case_id):
        m, k, b, Q, R, λs, dt, H, Qs, Rs, x0, model_name = defaultSetup(evaluation_config)
        self.mmae = MMAE(λs, dt, H, Q, R, x0, False, model_name)
        self.model_variants = λs
        self.case_id = case_id
        self.config = evaluation_config
    
    def run(self, dts, us, zs):
        print("Running Evaluation...")
        run_post = []
        run_λ_hat = []
        run_pdvs = []

        for dt, u, z in zip(dts, us, zs):
            λ_hat, cumulative_posteriors, pdvs = self.mmae.update(u, z, dt)
            print('Current measurement z:', z)
            print('Current estimated λs:', λ_hat)
            run_post.append(cumulative_posteriors)
            run_λ_hat.append(λ_hat)
            run_pdvs.append(pdvs)
            
        return array(run_λ_hat), array(run_post), array(run_pdvs)

    def defaultEvaluation(self, ts, dts, zs, us, store=True):
        # Labels necessary for final graphs file path, and title
        label_lambda = [f"./graphs/{self.case_id}_estimations.fig",
                        f'{self.case_id}: Parameter Estimates (m, k, b) vs Time']
        label_likely = [f"./graphs/{self.case_id}_likelyhoods.fig",
                        f'{self.case_id}: Heatmap of Model Likelihood Over Time']
        label_poster = [f"./graphs/{self.case_id}_posteriors.fig",
                        f'{self.case_id}: Heatmap of Cumulative Posterior Probabilities Over Time']

        run_λ_hat, run_post, run_pdvs = self.run(dts, us, zs)
        plotLambdaHat(ts, run_λ_hat, [self.config["true_m"], self.config["true_k"], self.config["true_b"]], label_lambda, store=store)
        plotHeatmap(run_pdvs, ts, self.model_variants, label_likely, store=store)
        plotHeatmap(run_post, ts, self.model_variants, label_poster, store=store)

if __name__ == "__main__":
    from computer_vision.tools.dataloader import Dataloader
    
    def testbenchEvaluation(evaluation_config, case_id, highdim: bool = False):
        dataloader = Dataloader("./output/")
        evaluation = Evaluation(evaluation_config, case_id)
        
        if highdim:
            ts, dts, cxs, cys, widths, heights = dataloader.load(f"./data/{case_id[:-3]}.csv")
            zs = [[[a], [b]] for a, b in zip((mean(cxs) - cxs), (mean(cys) - cys))]
            us = zeros_like(zs)
        else:
            ts, dts, cxs, cys, widths, heights = dataloader.load(f"./data/{case_id}.csv")
            zs = [[[a]] for a in (mean(cys) - cys)]
            us = zeros_like(zs)
        
        evaluation.defaultEvaluation(ts, dts, zs, us, True)
        plt.show()
    
    case_id = 'm095_0_k80_80'    
    evaluation_config_path = "./configuration_files/evaluation_configs/evaluation_m095_0_k80_80.json"
    evaluation_config = loadConfig(evaluation_config_path)
    testbenchEvaluation(evaluation_config, case_id, highdim=False)

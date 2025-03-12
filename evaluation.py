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
        self.dt = dt
        self.x0 = x0
        self.H = H
        self.R = R
        self.model_name = model_name
        self.q_variants = Qs
    
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

    def resetMMAE(self, Q):
        """
        Reset the MMAE algorithm state with a new Q matrix.
        """
        self.mmae = MMAE(self.model_variants, self.dt, self.H, Q, self.R, self.x0, False, self.model_name)
    
    def computeScore(self, run_λ_hat):
        # Compute the Mean Absolute Error (MMAE) based on the true values
        true_values = [self.config["true_m"], self.config["true_k"], self.config["true_b"]]  # Adjust as needed
        mmae_score = mean(abs(run_λ_hat - true_values))  # Example logic, adjust based on your needs
        return mmae_score
    
    def testQ(self, dts, us, zs):
        best_q = None
        best_mmae_score = float('inf')  # Initialize to a very large number
        
        for Q in self.q_variants:
            self.resetMMAE(Q)
            run_λ_hat, run_post, run_pdvs = self.run(dts, us, zs)
            # Compute MMAE score based on the results (you may need to adapt this based on your evaluation metric)
            mmae_score = self.computeScore(run_λ_hat)
            
            if mmae_score < best_mmae_score:
                best_mmae_score = mmae_score
                best_q = Q
        
        return best_q, best_mmae_score
    

    
if __name__ == "__main__":
    from computer_vision.tools.dataloader import Dataloader
    
    def testbenchEvaluation(evaluation_config, case_id, highdim: bool = False):
        dataloader = Dataloader("./output/")
        evaluation = Evaluation(evaluation_config, case_id)
        
        if highdim:
            ts, dts, cxs, cys, widths, heights = dataloader.load(f"./output/{case_id[:-3]}.csv")
            zs = [[[a], [b]] for a, b in zip((mean(cxs) - cxs), (mean(cys) - cys))]
            us = zeros_like(zs)
        else:
            ts, dts, cxs, cys, widths, heights = dataloader.load(f"./output/{case_id}.csv")
            zs = [[[a]] for a in (mean(cys) - cys)]
            us = zeros_like(zs)
        
        evaluation.defaultEvaluation(ts, dts, zs, us, True)
        plt.show()
    
    def testbenchTestQ(evaluation_config, case_id, highdim: bool = False, start_idx: int = 0, end_idx: int = None):
        dataloader = Dataloader("./output/")
        evaluation = Evaluation(evaluation_config, case_id)
        
        if highdim:
            ts, dts, cxs, cys, widths, heights = dataloader.load(f"./output/{case_id[:-3]}.csv")
            
            ts = ts[start_idx:end_idx]
            cxs = cxs[start_idx:end_idx]
            cys = cys[start_idx:end_idx]
            
            t = ts - ts[0]
            x = (mean(cxs) - cxs)
            y = (mean(cys) - cys)
            widths = widths[start_idx:end_idx]
            heights = heights[start_idx:end_idx]
            
            zs = [[[a], [b]] for a, b in zip(x, y)]
            us = zeros_like(zs)
        else:
            ts, dts, cxs, cys, widths, heights = dataloader.load(f"./output/{case_id}.csv")
            
            ts = ts[start_idx:end_idx]
            cxs = cxs[start_idx:end_idx]
            cys = cys[start_idx:end_idx]
            widths = widths[start_idx:end_idx]
            heights = heights[start_idx:end_idx]
            
            t = ts - ts[0]
            x = (mean(cxs) - cxs)
            y = (mean(cys) - cys)
            
            zs = [[[a]] for a in y]
            us = zeros_like(zs)
        
        detectionGraphics(case_id, t, x, y, widths, heights, False)
        plt.show()
        best_q, best_mmae_score = evaluation.testQ(dts, us, zs)
        print(f"Best Q matrix: {best_q}")
        print(f"Best MMAE score: {best_mmae_score}")
        
        # Proceed with the default evaluation using the best Q matrix (this assumes `defaultEvaluation` uses the best Q)
        evaluation.defaultEvaluation(ts, dts, zs, us, store=True)
        plt.show()
    
    case_id = 'sport_nopass_rb_norm'    
    evaluation_config_path = "./configuration_files/evaluation_configs/sport/evaluation_sport_nopass.json"
    evaluation_config = loadConfig(evaluation_config_path)
    testbenchTestQ(evaluation_config, case_id, highdim=False, start_idx=7)

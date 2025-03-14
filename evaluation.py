import sys
from numpy import array, mean, ndarray, empty, diag, zeros_like, interp
from scipy.interpolate import interp1d
from parameter_estimation_pipeline.MMAE.mmae import MMAE
from computer_vision.tools.common import *

class Evaluation:
    mmae: MMAE
    model_variants: list[ndarray]

    def __init__(self, evaluation_config, case_id):
        m, k, b, Q, R, λs, dt, H, Qs, Rs, x0, model_name = defaultSetup(evaluation_config)
        # print(Q)
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
    def determineMeasurementComp(file_path, highdim: bool = False):
        if highdim:
            data = loadtxt(file_path, delimiter=',', skiprows=1)
            ts, dts, x, y = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
            
            zs = [[[a], [b]] for a, b in zip(x, y)]
            us = zeros_like(zs)
        else:
            data = loadtxt(file_path, delimiter=',', skiprows=1)
            ts, dts, y = data[:, 0], data[:, 1], data[:, 3]
            
            zs = [[[a]] for a in y]
            us = zeros_like(zs)
            print(us)
            
        return ts, dts, zs, us
    
    
    # def determineMeasurementComp(file_path, inter=None, highdim: bool = False):
    #     data = loadtxt(file_path, delimiter=',', skiprows=1)
    #     ts, dts = data[:, 0], data[:, 1]

    #     if highdim:
    #         x, y = data[:, 2], data[:, 3]
    #         zs = [[[a], [b]] for a, b in zip(x, y)]
    #     else:
    #         y = data[:, 3]
    #         zs = [[[a]] for a in y]

    #     # If combined_ts is provided, interpolate data to match it
    #     if inter is not None:
    #         zs_interpolated = interp(inter, ts, array([z[0][0] for z in zs]))

    #         if highdim:
    #             zs = [[[a], [b]] for a, b in zip(zs_interpolated, zs_interpolated)]
    #         else:
    #             zs = [[[a]] for a in zs_interpolated]

    #         ts = inter  # Replace original timestamps with the aligned version

    #     us = zeros_like(zs)  # Assuming `us` doesn't require interpolation
    #     return ts, dts, zs, us
    
    def testbenchEvaluation(evaluation_config, case_id, file_path, highdim: bool = False):
        evaluation = Evaluation(evaluation_config, case_id)
        
        ts, dts, zs, us = determineMeasurementComp(file_path, highdim)
        
        evaluation.defaultEvaluation(ts, dts, zs, us, True)
        plt.show()
    
    def testbenchTestQ(evaluation_config, case_id, file_path, highdim: bool = False):
        evaluation = Evaluation(evaluation_config, case_id)
        
        ts, dts, zs, us = determineMeasurementComp(file_path, highdim)
        
        best_q, best_mmae_score = evaluation.testQ(dts, us, zs)
        print(f"Best Q matrix: {best_q}")
        print(f"Best MMAE score: {best_mmae_score}")
        
        evaluation.resetMMAE(best_q)
        evaluation.defaultEvaluation(ts, dts, zs, us, store=True)
        plt.show()
    
    def testbenchTestU(evaluation_config, case_id, zs_file_path, us_file_path, highdim: bool = False):
        evaluation = Evaluation(evaluation_config, case_id)
        
        ts_zs, dts_zs, zs, _ = determineMeasurementComp(zs_file_path, highdim)
        ts_us, dts_us, us, _ = determineMeasurementComp(us_file_path, highdim)
        
        plt.figure(figsize=(10, 6))
        plt.plot(ts_zs, [z[0][0] for z in zs], label='zs Data', marker='o')
        plt.plot(ts_us, [u[0][0] for u in us], label='us Data', marker='x')
        plt.legend()
        plt.title("Data Visualization")
        plt.xlabel("Time (s)")
        plt.ylabel("Measurement Values")
        plt.grid(True)

        # Interpolating missing points
        f_zs = interp1d(ts_zs, [z[0][0] for z in zs], kind='linear', fill_value='extrapolate')
        aligned_us = [f_zs(t) for t in ts_us]

        # Visualizing interpolated data
        plt.plot(ts_us, aligned_us, label='Interpolated zs Data (Aligned to us)', linestyle='--')
        # plt.plot(ts_us, [u[0][0] for u in us], label='Original us Data', marker='x')
        plt.legend()
        plt.title("Interpolated Data Visualization")
        plt.xlabel("Time (s)")
        plt.ylabel("Measurement Values")
        plt.grid(True)
        plt.show()

        
        # evaluation.defaultEvaluation(ts_zs, dts_zs, zs, us, True)
    
    case_id = 'sport_onepass_apriltag_tag_6_uin'
    zs_file_path = "./data/apriltag/sport_onepass_apriltag_tag_6.csv"
    us_file_path = "./data/apriltag/sport_onepass_apriltag_tag_5.csv"
    evaluation_config_path = "./configuration_files/evaluation_configs/apriltags/evaluation_sport_onepass_apriltag.json"
    evaluation_config = loadConfig(evaluation_config_path)
    
    # testbenchTestQ(evaluation_config, case_id, file_path, highdim=True)
    # testbenchEvaluation(evaluation_config, case_id, file_path, False)
    testbenchTestU(evaluation_config, case_id, zs_file_path, us_file_path, False)

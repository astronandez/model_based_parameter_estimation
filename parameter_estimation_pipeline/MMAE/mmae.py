from numpy import ndarray
import numpy as np
import matplotlib.pyplot as plt

from MMAE.Estimator_Likelihood.estimator_likelihood import EstimatorLikelihood
from MMAE.Joint_Probability.joint_probability import JointProbability
from System.system_simulator import SystemSimulator

class MMAE:
    def __init__(self, λs, dt, H, Q, R, x0, noisy):
        # Store parameter sets
        self.λs = np.array(λs)

        # Estimator likelihood simulator initialization
        self.EstimatorLikelihoods = [
            EstimatorLikelihood(λ, dt, H, Q, R, x0, noisy) for λ in λs
        ]

        # Joint probability simulator initialization
        self.JointProbability = JointProbability(λs)


    def update(self, u: ndarray, z: ndarray, dt: float) -> float:
        # Update each likelihood estimator
        pdvs = [
            EstimatorLikelihood.update(u, z, dt) 
            for EstimatorLikelihood in self.EstimatorLikelihoods
        ]
        
        # Update joint probabilities and get weighted estimate
        λ_hat, cumulative_posteriors = self.JointProbability.update(pdvs, self.λs)

        # self.manage_models(λ_hat, cumulative_posteriors)

        return λ_hat, cumulative_posteriors, pdvs
    

    def manage_models(self, λ_hat: np.ndarray, cumulative_posteriors: np.ndarray):
        """
        Adjust parameters of low-posterior models to have closer to the estimated dynamics (λ_hat)
        """
        threshold = 0.01   # Posterior threshold
        alpha = 0.1        # Adjustment weight

        for i, posterior in enumerate(cumulative_posteriors):
            if posterior <= threshold:
                # Smoothly adjust parameters toward λ_hat
                old_lambda = self.λs[i].copy()  # Keep a copy for logging
                self.λs[i] = (1 - alpha) * self.λs[i] + alpha * λ_hat
                
                # Update the model's parameters
                self.EstimatorLikelihoods[i].Estimator.SystemSimulator.model.λ = self.λs[i]
                
                # print(f"Model {i} updated from {old_lambda} to {self.λs[i]} based on posterior {posterior}")

                break
        

########### Testbench ###########

def ensure_positive_semidefinite(matrix):
    symmetric_matrix = (matrix + matrix.T) / 2
    eigvals = np.linalg.eigvalsh(symmetric_matrix)
    min_eigval = min(eigvals)
    if min_eigval < 0:
        symmetric_matrix += np.eye(symmetric_matrix.shape[0]) * (-min_eigval + 1e-8)
    return symmetric_matrix

def run_mmae_simulation(Q_scale, R_scale, λ_true, λ_variants, num_simulations=10000, num_steps=100, dt=0.1):
    np.random.seed(42)
    k_values = np.random.uniform(1.0, 5.0, num_simulations)
    b_values = np.random.uniform(1.0, 5.0, num_simulations)
    H = np.array([[1, 0]])
    Q_values = [ensure_positive_semidefinite(np.eye(H.shape[1]) * Q_scale * np.random.uniform(0.01, 1.0)) for _ in range(num_simulations)]
    R_values = [ensure_positive_semidefinite(np.eye(H.shape[0]) * R_scale * np.random.uniform(0.01, 1.0)) for _ in range(num_simulations)]
    x0 = np.array([0.0, 0.0]).reshape(2, 1)
    u = np.array([5.0]).reshape(1, 1)

    weighted_estimates_over_time = []

    for i in range(num_simulations):
        k = k_values[i]
        b = b_values[i]
        Q = Q_values[i]
        R = R_values[i]

        # Instantiate the true model
        true_model = SystemSimulator(λ_true, k, b, dt, H, Q, R, x0, noisy=True)

        # Instantiate MMAE
        mmae = MMAE(λ_variants, k, b, dt, H, Q, R, x0, noisy=False)

        simulation_weighted_estimates = []

        for t in range(num_steps):
            x_true, z = true_model.update(u)
            weighted_mass_estimate = mmae.update(u, z)
            simulation_weighted_estimates.append(weighted_mass_estimate)

        weighted_estimates_over_time.append(simulation_weighted_estimates)

    return np.array(weighted_estimates_over_time)

def plot_average_weighted_estimates(weighted_estimates_over_time):
    num_steps = weighted_estimates_over_time.shape[1]

    averaged_weighted_estimates = np.mean(weighted_estimates_over_time, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(range(num_steps), averaged_weighted_estimates, label='Weighted Estimate')
    plt.axhline(y=25.0, color='r', linestyle='--', label='True λ = 25.0')
    plt.title(f'Average Weighted Estimate over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Weighted Mass Estimate')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    num_simulations = 10000  # Increase for more robust averaging
    num_steps = 300
    dt = 0.1

    # Define the true model parameter and variant parameters
    λ_true = 25.0
    λ_variants = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

    # Define the scales for Q and R
    Q_scale = 100.0
    R_scale = 0.01

    print("Running MMAE simulations")
    weighted_estimates_over_time = run_mmae_simulation(Q_scale, R_scale, λ_true, λ_variants, num_simulations, num_steps, dt)
    print("Completed MMAE simulations\n")

    print(f"MMAE class Monte Carlo tests ({num_simulations} simulations) completed.")

    # Plot average weighted estimates
    plot_average_weighted_estimates(weighted_estimates_over_time)
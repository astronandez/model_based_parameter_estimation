from numpy import ndarray
import numpy as np
import matplotlib.pyplot as plt

from .conditional_probability_update import ConditionalProbabilityUpdate
from .weighted_estimate import WeightedEstimate
from ...System.system_simulator import SystemSimulator
from ..Estimator_Likelihood.estimator_likelihood import EstimatorLikelihood

class JointProbability:
    def __init__(self, λs):
        # Estimator likelihood simulators initialization
        self.λs = λs

        # Conditional probability update initialization
        self.ConditionalProbabilityUpdate = ConditionalProbabilityUpdate(λs)
        
        # Weighted estimate initialization
        self.weighted_estimate = WeightedEstimate(λs)
        

    def update(self, pdvs: ndarray, λs: ndarray) -> float:
        self.λs = λs
        cumulative_posteriors = self.ConditionalProbabilityUpdate.update(pdvs, λs)
        λ_hat = self.weighted_estimate.update(cumulative_posteriors, self.λs)

        return λ_hat, cumulative_posteriors
    

########### Testbench ###########

def ensure_positive_semidefinite(matrix):
    symmetric_matrix = (matrix + matrix.T) / 2
    eigvals = np.linalg.eigvalsh(symmetric_matrix)
    min_eigval = min(eigvals)
    if min_eigval < 0:
        symmetric_matrix += np.eye(symmetric_matrix.shape[0]) * (-min_eigval + 1e-8)
    return symmetric_matrix

def run_joint_probability_simulation(Q_scale, R_scale, λ_true, λ_variants, num_simulations=100, num_steps=100, dt=0.1):
    # np.random.seed(42)
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

        # Instantiate model variants
        estimator_likelihoods = [EstimatorLikelihood(λ_variant, k, b, dt, H, Q, R, x0, noisy=False) for λ_variant in λ_variants]
        joint_probability = JointProbability(λ_variants)

        simulation_weighted_estimates = []

        for t in range(num_steps):
            x_true, z = true_model.update(u)
            pdvs = np.array([estimator.update(u, z) for estimator in estimator_likelihoods])
            weighted_mass_estimate = joint_probability.update(pdvs)
            simulation_weighted_estimates.append(weighted_mass_estimate)

        weighted_estimates_over_time.append(simulation_weighted_estimates)

    return np.array(weighted_estimates_over_time)

def plot_average_weighted_estimates(results):
    num_steps = results[0][1].shape[1]

    for title_suffix, weighted_estimates_over_time in results:
        averaged_weighted_estimates = np.mean(weighted_estimates_over_time, axis=0)

        plt.figure(figsize=(10, 6))
        plt.plot(range(num_steps), averaged_weighted_estimates, label='Weighted Estimate')
        plt.axhline(y=25.0, color='r', linestyle='--', label='True λ = 25.0')
        plt.title(f'Average Weighted Estimate over Time - {title_suffix}')
        plt.xlabel('Time Step')
        plt.ylabel('Weighted Mass Estimate')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    num_simulations = 100  # Increase for more robust averaging
    num_steps = 400
    dt = 0.1

    # Define the true model parameter and variant parameters
    λ_true = 25.0
    λ_variants = [20.0, 25.0, 30.0]

    # Define the scales for Q and R
    large_scale = 100.0
    small_scale = 0.01

    results = []

    # Run simulations for each combination of Q and R scales
    combinations = [
        ('large_Q_large_R', large_scale, large_scale),
        ('large_Q_small_R', large_scale, small_scale),
        ('small_Q_large_R', small_scale, large_scale),
        ('small_Q_small_R', small_scale, small_scale)
    ]

    for title_suffix, Q_scale, R_scale in combinations:
        print(f"Running simulations for {title_suffix}")
        weighted_estimates_over_time = run_joint_probability_simulation(Q_scale, R_scale, λ_true, λ_variants, num_simulations, num_steps, dt)
        results.append((title_suffix, weighted_estimates_over_time))
        print(f"Completed simulations for {title_suffix}\n")

    print(f"JointProbability class Monte Carlo tests ({num_simulations} simulations) completed for all Q and R combinations.")

    # Plot average weighted estimates for each combination of Q and R scales
    plot_average_weighted_estimates(results)
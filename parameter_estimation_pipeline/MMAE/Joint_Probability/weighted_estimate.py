from numpy import ndarray
import numpy as np
import matplotlib.pyplot as plt

from ...System.system_simulator import SystemSimulator
from ..Estimator_Likelihood.estimator_likelihood import EstimatorLikelihood
from .conditional_probability_update import ConditionalProbabilityUpdate

class WeightedEstimate:
    def __init__(self, λs: ndarray):
        """
        :param λs: An array where each element is a vector [m, k, b] representing model parameters.
        """
        self.λs = λs  # List of parameter vectors for each model (each λ = [m, k, b])

    def update(self, cumulative_posteriors: ndarray, λs: ndarray) -> ndarray:
        """
        Calculate the most likely model and the weighted estimates for each parameter (m, k, b).
        
        :param cumulative_posteriors: A 1D array of posterior probabilities for each model.
        :return: A vector of weighted parameter estimates [weighted_m, weighted_k, weighted_b].
        """
        self.λs = λs
        # Initialize a vector for the weighted estimate [m, k, b]
        weighted_estimates = np.array([0.0, 0.0, 0.0])

        # Calculate the weighted estimates for each parameter in the vector [m, k, b]
        for i in range(len(weighted_estimates)):
            weighted_estimates[i] = np.sum([p * λ[i] for p, λ in zip(cumulative_posteriors, self.λs)])

        return weighted_estimates
    

########### Testbench ###########

def ensure_positive_semidefinite(matrix):
    symmetric_matrix = (matrix + matrix.T) / 2
    eigvals = np.linalg.eigvalsh(symmetric_matrix)
    min_eigval = min(eigvals)
    if min_eigval < 0:
        symmetric_matrix += np.eye(symmetric_matrix.shape[0]) * (-min_eigval + 1e-8)
    return symmetric_matrix

def run_estimator_likelihood_simulation(Q_scale, R_scale, λ_true, λ_variants, num_simulations=100, num_steps=100, dt=0.1):
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

        # Instantiate model variants
        estimator_likelihoods = [EstimatorLikelihood(λ_variant, k, b, dt, H, Q, R, x0, noisy=False) for λ_variant in λ_variants]
        cond_prob_update = ConditionalProbabilityUpdate(λ_variants)
        weighted_estimate = WeightedEstimate(λ_variants)

        simulation_weighted_estimates = []

        for t in range(num_steps):
            x_true, z = true_model.update(u)
            pdvs = np.array([estimator.update(u, z) for estimator in estimator_likelihoods])
            model_probabilities = cond_prob_update.update(pdvs)
            weighted_mass_estimate = weighted_estimate.update(model_probabilities)
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
    λ_variants = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]

    # Define the scales for Q and R
    large_scale = 50.0
    small_scale = 0.1

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
        weighted_estimates_over_time = run_estimator_likelihood_simulation(Q_scale, R_scale, λ_true, λ_variants, num_simulations, num_steps, dt)
        results.append((title_suffix, weighted_estimates_over_time))
        print(f"Completed simulations for {title_suffix}\n")

    print(f"EstimatorLikelihood class Monte Carlo tests ({num_simulations} simulations) completed for all Q and R combinations.")

    # Plot average weighted estimates for each combination of Q and R scales
    plot_average_weighted_estimates(results)
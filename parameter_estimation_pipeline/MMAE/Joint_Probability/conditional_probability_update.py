from numpy import ndarray
import numpy as np
import matplotlib.pyplot as plt

from System.system_simulator import SystemSimulator
from ..Estimator_Likelihood.estimator_likelihood import EstimatorLikelihood

class ConditionalProbabilityUpdate:
    def __init__(self, λs, lower_bound=0.01, upper_bound=0.99):
        # Estimator likelihood simulators initialization
        self.λs = λs

        # Model probabilities initialization
        self.cumulative_posteriors = np.ones(len(self.λs)) / len(self.λs)
        
        # Threshold bounds for posterior probabilities
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def update(self, pdvs: np.ndarray, λs: np.ndarray) -> np.ndarray:
        """
        Update the model probabilities using Bayes' theorem with clipping to avoid collapse.
        """
        self.λs = λs
        self.cumulative_posteriors *= 0.6  # Decay previous posteriors

        # Calculate the marginal likelihood of z (the normalization factor):
        norm_factor = np.sum(pdvs * self.cumulative_posteriors) + 1.0e-40

        # Update the model probabilities (posterior probabilities) using Bayes' theorem:
        new_posteriors = (pdvs * self.cumulative_posteriors) / norm_factor

        # Apply clipping to ensure posteriors are within bounds
        # new_posteriors = np.clip(new_posteriors, self.lower_bound, self.upper_bound)

        # Renormalize posteriors to ensure they sum to 1 after clipping
        new_posteriors /= np.sum(new_posteriors)

        # Update cumulative posteriors
        self.cumulative_posteriors = new_posteriors

        return self.cumulative_posteriors
    

########### Testbench ###########

def ensure_positive_semidefinite(matrix):
    symmetric_matrix = (matrix + matrix.T) / 2
    eigvals = np.linalg.eigvalsh(symmetric_matrix)
    min_eigval = min(eigvals)
    if min_eigval < 0:
        symmetric_matrix += np.eye(symmetric_matrix.shape[0]) * (-min_eigval + 1e-8)
    return symmetric_matrix

def run_estimator_likelihood_simulation(Q_scale, R_scale, λ_true, λ_variants, num_simulations=100, num_steps=100, dt=0.1):
    # np.random.seed(42)
    k_values = np.random.uniform(1.0, 5.0, num_simulations)
    b_values = np.random.uniform(1.0, 5.0, num_simulations)
    H = np.array([[1, 0]])
    Q_values = [ensure_positive_semidefinite(np.eye(H.shape[1]) * Q_scale * np.random.uniform(0.01, 1.0)) for _ in range(num_simulations)]
    R_values = [ensure_positive_semidefinite(np.eye(H.shape[0]) * R_scale * np.random.uniform(0.01, 1.0)) for _ in range(num_simulations)]
    x0 = np.array([0.0, 0.0]).reshape(2, 1)
    u = np.array([5.0]).reshape(1, 1)

    probabilities_over_time = []

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

        simulation_probabilities = []

        for t in range(num_steps):
            x_true, z = true_model.update(u)
            pdvs = np.array([estimator.update(u, z) for estimator in estimator_likelihoods])
            model_probabilities = cond_prob_update.update(pdvs)
            simulation_probabilities.append(model_probabilities)

        probabilities_over_time.append(simulation_probabilities)

    return np.array(probabilities_over_time)

def plot_averaged_heatmaps(results, λ_variants):
    num_steps = results[0][1].shape[1]
    num_models = len(λ_variants)

    for title_suffix, probabilities_over_time in results:
        averaged_probabilities = np.mean(probabilities_over_time, axis=0)

        plt.figure(figsize=(10, 6))
        plt.imshow(averaged_probabilities.T, aspect='auto', cmap='hot', interpolation='nearest')
        plt.colorbar(label='Probability')
        plt.title(f'Averaged Heatmap of Model Probabilities over Time - {title_suffix}')
        plt.xlabel('Time Step')
        plt.ylabel('Model Index')
        plt.yticks(ticks=range(num_models), labels=[f'λ = {λ}' for λ in λ_variants])
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
        probabilities_over_time = run_estimator_likelihood_simulation(Q_scale, R_scale, λ_true, λ_variants, num_simulations, num_steps, dt)
        results.append((title_suffix, probabilities_over_time))
        print(f"Completed simulations for {title_suffix}\n")

    print(f"EstimatorLikelihood class Monte Carlo tests ({num_simulations} simulations) completed for all Q and R combinations.")

    # Plot averaged heatmaps for each combination of Q and R scales
    plot_averaged_heatmaps(results, λ_variants)
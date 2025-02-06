import os
import numpy as np
from numpy import ndarray, eye, mean, var, eye
import matplotlib.pyplot as plt
from numpy.linalg import eigvalsh, inv
from scipy.stats import shapiro, kstest

from ....System.system_simulator import SystemSimulator
from .estimator_update import EstimatorUpdate

class Estimator:
    def __init__(self, λ, dt, H, Q, R, x0, noisy):
        # Dynamic System Simulator initialization for state estimator
        self.SystemSimulator = SystemSimulator(λ, dt, H, Q, R, x0, noisy)

        # state Estimator update initialization
        self.EstimatorUpdate = EstimatorUpdate(self.SystemSimulator.model, eye(self.SystemSimulator.model.Φ.shape[1]), x0)

        # Initialize the state estimate output from the update phase. this will be x̂(t-1)
        self.x̂_ = None
    
    
    def update(self, u: ndarray, z: ndarray, dt: float):
        # State estimate before the measurement update phase
        _x̂, ẑ = self.SystemSimulator.update(u, dt, self.x̂_)

        # State estimate after the measurement update phase
        x̂, P, r, A = self.EstimatorUpdate.update(z, _x̂, ẑ)

        # Update the state estimate output from the update phase. this will be x̂(t-1) in next iteration
        self.x̂_ = x̂
        
        return x̂, P, r, A
    

########### Testbench ###########

# Ensure positive semidefinite matrix
def ensure_positive_semidefinite(matrix):
    symmetric_matrix = (matrix + matrix.T) / 2
    eigvals = eigvalsh(symmetric_matrix)
    min_eigval = min(eigvals)
    if min_eigval < 0:
        symmetric_matrix += np.eye(symmetric_matrix.shape[0]) * (-min_eigval + 1e-8)
    return symmetric_matrix

def validate_residuals(residuals, A_matrices):
    residuals_flattened = np.concatenate(residuals)
    A_matrices_mean = np.mean(A_matrices, axis=0)

    # Check mean of residuals
    residuals_mean = np.mean(residuals_flattened, axis=0)
    print(f"Mean of residuals: {residuals_mean}")

    # Check covariance of residuals
    residuals_covariance = np.cov(residuals_flattened.T)
    print(f"Covariance of residuals: \n{residuals_covariance}")
    print(f"Mean A matrix: \n{A_matrices_mean}")

    # Perform normality test
    stat, p_value = shapiro(residuals_flattened)
    print(f"Shapiro-Wilk Test: Statistics={stat}, p-value={p_value}")
    if p_value > 0.05:
        print("Residuals are normally distributed")
    else:
        print("Residuals are not normally distributed")

    # Kolmogorov-Smirnov test for goodness-of-fit to a normal distribution
    stat, p_value = kstest(residuals_flattened, 'norm', args=(residuals_mean, np.sqrt(np.diag(A_matrices_mean))))
    print(f"Kolmogorov-Smirnov Test: Statistics={stat}, p-value={p_value}")
    if p_value > 0.05:
        print("Residuals fit the normal distribution")
    else:
        print("Residuals do not fit the normal distribution")

    # Check for White Noise Characteristics
    # lbvalue, pvalues, *_ = acorr_ljungbox(residuals_flattened, lags=[10], return_df=False)
    # pvalue = float(pvalues[0])
    # print(f"Ljung-Box Test: Statistics={lbvalue}, p-value={pvalue}")
    # if pvalue > 0.05:
    #     print("Residuals are white (uncorrelated)")
    # else:
    #     print("Residuals are not white (correlated)")


def run_estimator_simulation(Q_scale, R_scale, num_simulations=1000, num_steps=100, dt=0.1):
    np.random.seed(42)  # For reproducibility
    λ_values = np.random.uniform(10.0, 50.0, num_simulations)
    k_values = np.random.uniform(1.0, 5.0, num_simulations)
    b_values = np.random.uniform(1.0, 5.0, num_simulations)
    H = np.array([[1, 0]])  # Fixed H
    Q_values = [ensure_positive_semidefinite(np.eye(H.shape[1]) * Q_scale * np.random.uniform(0.01, 1.0)) for _ in range(num_simulations)]
    R_values = [ensure_positive_semidefinite(np.eye(H.shape[0]) * R_scale * np.random.uniform(0.01, 1.0)) for _ in range(num_simulations)]

    all_true_states = []
    all_estimated_states = []
    all_measurements = []
    mse_values = []
    mean_errors = []
    var_errors = []

    u = np.array([5.0]).reshape(1, 1)  # Control input

    for i in range(num_simulations):
        λ = λ_values[i]
        k = k_values[i]
        b = b_values[i]
        Q = Q_values[i]
        R = R_values[i]
        x0 = np.array([0.0, 0.0]).reshape(2, 1)  # Initial state

        # Initialize SystemSimulator for the actual plant (noisy)
        plant_simulator = SystemSimulator(λ, k, b, dt, H, Q, R, x0, noisy=True)

        # Initialize Estimator
        estimator = Estimator(λ, k, b, dt, H, Q, R, x0, noisy=False)

        true_states = []
        estimated_states = []
        measurements = []
        residuals = []
        A_matrices = []

        for t in range(num_steps):
            x_true, z = plant_simulator.update(u)
            x_estimated, P, r, A = estimator.update(u, z)
            z_hat = estimator.SystemSimulator.model.H @ x_estimated

            true_states.append(x_true.flatten())
            estimated_states.append(z_hat.flatten())
            measurements.append(z.flatten())
            residuals.append(r.flatten())
            A_matrices.append(A)

        all_true_states.append(np.array(true_states))
        all_estimated_states.append(np.array(estimated_states))
        all_measurements.append(np.array(measurements))

        validate_residuals(residuals, A_matrices)

        mse = np.mean(np.square(np.array(true_states)[:, 0] - np.array(estimated_states)[:, 0]))
        mse_values.append(mse)
        mean_errors.append(np.mean(np.array(true_states) - np.array(estimated_states), axis=0))
        var_errors.append(np.var(np.array(true_states) - np.array(estimated_states), axis=0))

    return all_true_states, all_estimated_states, all_measurements, mse_values, mean_errors, var_errors

# Plot combined results
def plot_combined_results(results, dt):
    time = np.arange(num_steps) * dt

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    for ax, (title_suffix, all_measurements, all_estimated_states, mse_values) in zip(axs.flat, results):
        # Plot a subset of the simulations for readability
        for i in range(0, num_simulations, num_simulations // 10):
            ax.plot(time, all_measurements[i][:, 0], alpha=0.6, linestyle='--', label=f'Measured State {i}' if i == 0 else "")
            ax.plot(time, all_estimated_states[i][:, 0], alpha=0.6, label=f'Estimated State {i}' if i == 0 else "")
        ax.set_title(f'{title_suffix}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('State (z and ẑ)')
        if i == 0:
            ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join('output', 'estimator_simulation_combined.png'))
    plt.show()
    plt.close()

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    for ax, (title_suffix, all_measurements, all_estimated_states, mse_values) in zip(axs.flat, results):
        ax.hist(mse_values, bins=50, alpha=0.75, label=f'MSE {title_suffix}')
        ax.set_title(f'{title_suffix}')
        ax.set_xlabel('MSE')
        ax.set_ylabel('Frequency')
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join('output', 'estimator_simulation_mse_combined.png'))
    plt.show()
    plt.close()

if __name__ == "__main__":
    num_simulations = 1000
    num_steps = 200
    dt = 0.1

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
        all_true_states, all_estimated_states, all_measurements, mse_values, mean_errors, var_errors = run_estimator_simulation(Q_scale, R_scale, num_simulations, num_steps, dt)
        results.append((title_suffix, all_measurements, all_estimated_states, mse_values))
    
    plot_combined_results(results, dt)
    
    print(f"Estimator class Monte Carlo tests ({num_simulations} simulations) completed for all Q and R combinations.")
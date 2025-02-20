import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import os

from ..System.Model.models import MultivariableSimpleHarmonicOscillator, MultivariableSimpleHarmonicOscillator2D
from ..System.plant import Plant

class SystemSimulator:
    def __init__(self, λ, dt, H, Q, R, x0, noisy, model_name="MultivariableSimpleHarmonicOscillator"):
        # Model initialization
        if model_name in globals():
            self.model = globals()[model_name](λ, dt, H, Q, R)
        else:
            raise ValueError(f"Model '{model_name}' not found.")

        # Plant initialization
        self.plant = Plant(self.model, x0, noisy)


    def update(self, u: ndarray, dt: float, x̂_: ndarray = None) -> ndarray:
        self.model.update_dt(dt)
        return self.plant.update(u, x̂_)
    

########### Testbench ###########

def ensure_positive_semidefinite(matrix: np.ndarray) -> np.ndarray:
    """Ensure that a matrix is positive semidefinite."""
    symmetric_matrix = (matrix + matrix.T) / 2
    eigvals = np.linalg.eigvalsh(symmetric_matrix)
    min_eigval = min(eigvals)
    if min_eigval < 0:
        symmetric_matrix += np.eye(symmetric_matrix.shape[0]) * (-min_eigval + 1e-8)
    return symmetric_matrix

# Run simulations
def run_simulation(Q_scale, R_scale, num_simulations=1000, num_steps=100, dt=0.1):
    np.random.seed(42)  # For reproducibility
    λ_values = np.random.uniform(10.0, 50.0, num_simulations)
    k_values = np.random.uniform(1.0, 5.0, num_simulations)
    b_values = np.random.uniform(1.0, 5.0, num_simulations)
    H = np.array([[1, 0]])  # Fixed H
    Q_values = [ensure_positive_semidefinite(np.eye(H.shape[1]) * Q_scale * np.random.uniform(0.01, 1.0)) for _ in range(num_simulations)]
    R_values = [ensure_positive_semidefinite(np.eye(H.shape[0]) * R_scale * np.random.uniform(0.01, 1.0)) for _ in range(num_simulations)]

    all_outputs = []
    all_states = []
    mse_values = []

    for i in range(num_simulations):
        λ = λ_values[i]
        k = k_values[i]
        b = b_values[i]
        Q = Q_values[i]
        R = R_values[i]
        x0 = np.array([0.0, 0.0]).reshape(2, 1)  # Initial state
        u = np.array([5.0]).reshape(1, 1)  # Control input

        # Initialize SystemSimulator
        simulator = SystemSimulator(λ, k, b, dt, H, Q, R, x0, noisy=True)

        outputs = []
        states = []
        for t in range(num_steps):
            x, z = simulator.update(u)
            outputs.append(z.flatten())
            states.append(x.flatten())

        all_outputs.append(np.array(outputs))
        all_states.append(np.array(states))
        states = np.array(states)
        outputs = np.array(outputs)
        
        mse = np.mean(np.square(states[:, 0] - outputs[:, 0]))  # Compare the first state variable with the output
        mse_values.append(mse)

    return all_states, all_outputs, mse_values

# Plot combined results
def plot_combined_results(results, dt):
    time = np.arange(num_steps) * dt

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    for ax, (title_suffix, all_states, all_outputs, mse_values) in zip(axs.flat, results):
        # Plot a subset of the simulations for readability
        for i in range(0, num_simulations, num_simulations // 10):
            ax.plot(time, all_states[i][:, 0], alpha=0.6, linestyle='--', label=f'State {i}' if i == 0 else "")
            ax.plot(time, all_outputs[i][:, 0], alpha=0.6, label=f'Output {i}' if i == 0 else "")
        ax.set_title(f'{title_suffix}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('State (x) and Output (z)')
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join('output', 'system_simulator_simulation_combined.png'))
    plt.show()
    plt.close()

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    for ax, (title_suffix, all_states, all_outputs, mse_values) in zip(axs.flat, results):
        ax.hist(mse_values, bins=50, alpha=0.75, label=f'MSE {title_suffix}')
        ax.set_title(f'{title_suffix}')
        ax.set_xlabel('MSE')
        ax.set_ylabel('Frequency')
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join('output', 'system_simulator_simulation_mse_combined.png'))
    plt.show()
    plt.close()

if __name__ == "__main__":
    num_simulations = 1000
    num_steps = 200
    dt = 0.1

    # Define the scales for Q and R
    large_scale = 50.0
    small_scale = 0.1

    # Run simulations for each combination of Q and R scales
    combinations = [
        ('large_Q_large_R', large_scale, large_scale),
        ('large_Q_small_R', large_scale, small_scale),
        ('small_Q_large_R', small_scale, large_scale),
        ('small_Q_small_R', small_scale, small_scale)
    ]

    results = []
    for title_suffix, Q_scale, R_scale in combinations:
        all_states, all_outputs, mse_values = run_simulation(Q_scale, R_scale, num_simulations, num_steps, dt)
        results.append((title_suffix, all_states, all_outputs, mse_values))
    
    plot_combined_results(results, dt)
    
    print(f"SystemSimulator class Monte Carlo tests ({num_simulations} simulations) completed for all Q and R combinations.")
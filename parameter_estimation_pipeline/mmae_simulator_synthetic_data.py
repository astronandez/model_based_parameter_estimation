import numpy as np

from inputs import Input
from MMAE.mmae import MMAE
from System.system_simulator import SystemSimulator
from testbench_tools import simulation_configuration_setup
from testbench_tools import mmae_simulator_plots

class MMAESimulatorSyntheticData:
    def __init__(self, λs, true_λ, dt, H, Q_mmae, R_mmae, Q_true_system, R_true_system, x0, true_system_noisy, estimator_noisy, max_time, max_steps, amplitude):
        # Synthetic system simulator initialization
        self.TrueSystem = SystemSimulator(true_λ, dt, H, Q_true_system, R_true_system, x0, true_system_noisy)

        # Input initialization
        self.input_signal = Input(self.TrueSystem.model, max_time).step_function(max_steps, amplitude)

        # MMAE initialization
        self.MMAE = MMAE(λs, dt, H, Q_mmae, R_mmae, x0, estimator_noisy)


    def update(self, t: int, dt: float) -> float:
        u = self.input_signal[t, :].reshape(-1, 1)
        _, z = self.TrueSystem.update(u, dt)
        zs.append(z.flatten())  # Appends a 1D array instead of 2D
        λ_hat, cumulative_posteriors, pdvs = self.MMAE.update(u, z, dt)

        return λ_hat, cumulative_posteriors, pdvs


if __name__ == "__main__":
    # Load configuration from JSON file
    λs, m, k, b, dt, H, Q_mmae, R_mmae, Q_true_system, R_true_system, x0, max_time, max_steps, amplitude, random_seed, true_system_noisy = simulation_configuration_setup("../configuration_methods/parameter_estimation_configs/config_simulated_data.json")

    # Set random seed
    np.random.seed(random_seed)

    true_λ = np.array([m, k, b])

    # MMAE simulator initialization
    MMAESimulator = MMAESimulatorSyntheticData(λs, true_λ, dt, H, Q_mmae, R_mmae, Q_true_system, R_true_system, x0, true_system_noisy, False, max_time, max_steps, amplitude)

    # Lists to store time and λ_hat values
    times = []
    lambda_hats = []
    zs = []
    cumulative_posteriors_summary = []
    pdvs_summary = []

    # Main simulation loop
    for step_counter in range(0, max_steps):
        λ_hat, cumulative_posteriors, pdvs = MMAESimulator.update(step_counter, dt)
        times.append(step_counter * dt)
        lambda_hats.append(λ_hat)
        cumulative_posteriors_summary.append(cumulative_posteriors)
        pdvs_summary.append(pdvs)

    mmae_simulator_plots(times, true_λ, λs, zs, lambda_hats, cumulative_posteriors_summary, pdvs_summary)

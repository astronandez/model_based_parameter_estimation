import numpy as np

from inputs import Input
from MMAE.mmae import MMAE
from System.system_simulator import SystemSimulator
from testbench_tools import simulation_configuration_setup
from testbench_tools import mmae_simulator_plots

class MMAESimulatorRealData:
    def __init__(self, λs, true_λ, dt, H, Q, R, x0, true_system_noisy, estimator_noisy, max_time, max_steps, amplitude):
        # Synthetic system simulator initialization
        self.TrueSystem = SystemSimulator(true_λ, dt, H, Q, R, x0, true_system_noisy)

        # Input initialization
        self.input_signal = Input(self.TrueSystem.model, max_time).step_function(max_steps, amplitude)

        # MMAE initialization
        self.MMAE = MMAE(λs, dt, H, Q, R, x0, estimator_noisy)
    
    def update(self, t: int, z: np.ndarray, dt: float) -> float:
        u = self.input_signal[t, :].reshape(-1, 1)

        λ_hat, cumulative_posteriors, pdvs = self.MMAE.update(u, z, dt)

        return λ_hat, cumulative_posteriors, pdvs


if __name__ == "__main__":
    # Load configuration from JSON file
    λs, m, k, b, dt, H, Q_mmae, R_mmae, Q_true_system, R_true_system, x0, max_time, max_steps, amplitude, random_seed, true_system_noisy = simulation_configuration_setup("config_real_data.json")

    # Set random seed
    np.random.seed(random_seed)

    true_λ = np.array([m, k, b])

    # MMAE simulator initialization
    MMAESimulator = MMAESimulatorRealData(λs, true_λ, dt, H, Q_mmae, R_mmae, x0, True, False, max_time, max_steps, amplitude)

    ### Run simulation ###
    times = []
    zs = []
    lambda_hats = []
    cumulative_posteriors_summary = []
    pdvs_summary = []

    data = np.loadtxt("./Data/lemur_sticky_sport_load_measurements.csv", delimiter=",", usecols=4, skiprows=1)  # skiprows=1 if there is a header
    dts = np.loadtxt("./Data/lemur_sticky_sport_load_measurements.csv", delimiter=",", usecols=2, skiprows=1)  # skiprows=1 if there is a header

    # Init Initial state
    time_track = 0.0
    times.append(time_track)
    z = np.array([[data[0]]], dtype='float64')
    zs.append(z)
    λ_hat, cumulative_posteriors, pdvs = MMAESimulator.update(0, z, dt)
    lambda_hats.append(λ_hat)
    cumulative_posteriors_summary.append(cumulative_posteriors)
    pdvs_summary.append(pdvs)

    # Main simulation loop
    for step_counter in range(1, max_time):
        dt = dts[step_counter]
        time_track += dt
        times.append(time_track)
        z = np.array([[data[step_counter]]], dtype='float64')
        zs.append(z)
        λ_hat, cumulative_posteriors, pdvs = MMAESimulator.update(step_counter, z, dt)
        lambda_hats.append(λ_hat)
        cumulative_posteriors_summary.append(cumulative_posteriors)
        pdvs_summary.append(pdvs)

    # Convert zs to a 2D array after the loop
    zs = np.array(zs).reshape(len(zs), -1)

    mmae_simulator_plots(times, true_λ, λs, zs, lambda_hats, cumulative_posteriors_summary, pdvs_summary)

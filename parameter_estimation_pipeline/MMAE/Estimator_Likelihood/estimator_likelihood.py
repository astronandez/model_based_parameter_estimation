import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray, eye, mean, var
from numpy.linalg import eigvalsh, inv
from scipy.stats import shapiro, kstest, norm, ttest_ind, multivariate_normal
from statsmodels.stats.diagnostic import acorr_ljungbox
 
from .Estimator.estimator import Estimator
from .PDV.pdv import PDV
from System.system_simulator import SystemSimulator
from .Spectral_Estimator.spectral_estimator import SpectralEstimator

class EstimatorLikelihood:
    def __init__(self, λ, dt, H, Q, R, x0, noisy):
        # State estimator initialization
        self.Estimator = Estimator(λ, dt, H, Q, R, x0, noisy)

        # Spectral estimator initialization
        self.SpectralEstimator = SpectralEstimator(50, 1)

        # Compute scalar likelihood initialization
        self.PDV = PDV()


    def update(self, u: ndarray, z: ndarray, dt: float):
        _, _, r, A = self.Estimator.update(u, z, dt)
        r_k_gamma, A_k_gamma = self.SpectralEstimator.update(r)
        pdv = self.PDV.update(r_k_gamma, A_k_gamma)
        # pdv = self.PDV.update(r, A)

        return pdv
    

########### Testbench ###########

def ensure_positive_semidefinite(matrix):
    symmetric_matrix = (matrix + matrix.T) / 2
    eigvals = np.linalg.eigvalsh(symmetric_matrix)
    min_eigval = min(eigvals)
    if min_eigval < 0:
        symmetric_matrix += np.eye(symmetric_matrix.shape[0]) * (-min_eigval + 1e-8)
    return symmetric_matrix

def run_estimator_likelihood_simulation(Q_scale, R_scale, num_simulations=100, num_steps=100, dt=0.1):
    np.random.seed(42)  # For reproducibility
    λ_values = np.random.uniform(10.0, 50.0, num_simulations)
    k_values = np.random.uniform(1.0, 5.0, num_simulations)
    b_values = np.random.uniform(1.0, 5.0, num_simulations)
    H = np.array([[1, 0]])  # Fixed H
    Q_values = [ensure_positive_semidefinite(np.eye(H.shape[1]) * Q_scale * np.random.uniform(0.01, 1.0)) for _ in range(num_simulations)]
    R_values = [ensure_positive_semidefinite(np.eye(H.shape[0]) * R_scale * np.random.uniform(0.01, 1.0)) for _ in range(num_simulations)]

    pdv_values = []
    u = np.array([5.0]).reshape(1, 1)  # Control input

    for i in range(num_simulations):
        λ = λ_values[i]
        k = k_values[i]
        b = b_values[i]
        Q = Q_values[i]
        R = R_values[i]
        x0 = np.array([0.0, 0.0]).reshape(2, 1)  # Initial state

        plant_simulator = SystemSimulator(λ, k, b, dt, H, Q, R, x0, noisy=True)
        estimator_likelihood = EstimatorLikelihood(λ, k, b, dt, H, Q, R, x0, noisy=False)

        pdv_simulation_values = []
        rs = []
        As = []

        for t in range(num_steps):
            x_true, z = plant_simulator.update(u)
            _, _, r, A = estimator_likelihood.Estimator.update(u, z)
            pdv = estimator_likelihood.PDV.update(r, A)

            rs.append(r)
            As.append(A)
            pdv_simulation_values.append(pdv)
        
        pdv_values.append((pdv_simulation_values, rs, As))

    return pdv_values

def plot_pdv_distributions(pdv_values, num_steps, num_simulations):
    for sim in range(num_simulations):
        pdv_simulation_values, rs, As = pdv_values[sim]
        for t in range(num_steps):
            r = rs[t].flatten()  # Ensure r is a 1D array
            A = As[t].item()  # Ensure A is treated as a scalar
            pdv = pdv_simulation_values[t]
            
            mean = 0
            std_dev = np.sqrt(A)
            x = np.linspace(-3*std_dev, 3*std_dev, 1000)
            y = norm.pdf(x, mean, std_dev)
            
            fig, ax = plt.subplots()
            ax.plot(x, y, label='Normal Distribution')
            ax.plot(r, pdv, 'ro', label=f'Residual r: {r}\nPDV={pdv:.4f}')
            plt.title(f'1D Gaussian Distribution with Residual at Time Step {t+1} (Sim {sim+1})')
            plt.xlabel('Residual Value')
            plt.ylabel('Probability Density')
            plt.legend()
            plt.show()

if __name__ == "__main__":
    num_simulations = 10  # Number of runs
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
        print(f"Running simulations for {title_suffix}")
        pdv_values = run_estimator_likelihood_simulation(Q_scale, R_scale, num_simulations, num_steps, dt)
        results.append((title_suffix, pdv_values))
        print(f"Completed simulations for {title_suffix}\n")

    print(f"EstimatorLikelihood class Monte Carlo tests ({num_simulations} simulations) completed for all Q and R combinations.")
    
    # Plot PDV distributions for the first combination as an example
    title_suffix, pdv_values = results[0]
    plot_pdv_distributions(pdv_values, num_steps, num_simulations)
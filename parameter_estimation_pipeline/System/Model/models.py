from numpy import ndarray, array, mean, square, arange, linalg, eye, allclose
from numpy.random import seed, uniform
import matplotlib.pyplot as plt
import os

from .system import System

class SimpleHarmonicOscillator(System):
    def __init__(self, λ: float, k: float, b: float, dt: float, H: ndarray, Q: ndarray, R: ndarray):
        self.λ = λ  # Mass
        self.k = k  # Spring constant
        self.b = b  # Damping coefficient
        self.dt = dt  # Time step

        super().__init__(
                        # State transition matrix (A)
                        Φ = array([[1, dt],
                                   [- (k * dt) / λ, 1 - ((b / λ) * dt)]]),

                        # Input transition matrix (B)
                        B = array([[0],
                                   [dt/λ]]),
                        
                        H=H,
                        Q=Q,
                        R=R
        )

    def update_dt(self, dt: float):
        self.dt = dt
        self.update_matrices()
        
    def update_matrices(self):
        # Recalculate the state transition matrix Φ and the input transition matrix B
        self.Φ = array([[1, self.dt],
                        [- (self.k * self.dt) / self.λ, 1 - ((self.b / self.λ) * self.dt)]])
        self.B = array([[0],
                        [self.dt / self.λ]])

class MultivariableSimpleHarmonicOscillator(System):
    def __init__(self, λ: ndarray, dt: float, H: ndarray, Q: ndarray, R: ndarray):
        self.λ = array(λ)
        
        self.λ = λ      # Parameter vector
        self.m = λ[0]   # Mass
        self.k = λ[1]   # Spring constant
        self.b = λ[2]   # Damping coefficient
        self.dt = dt    # Time step

        super().__init__(
                        # State transition matrix (A)
                        Φ = array([[1, self.dt],
                                   [- (self.k * self.dt) / self.m, 1 - ((self.b / self.m) * self.dt)]]),

                        # Input transition matrix (B)
                        B = array([[0],
                                   [self.dt/self.m]]),
                        
                        H=H,
                        Q=Q,
                        R=R,
                        λ=self.λ
        )
    
    def update_dt(self, dt: float):
        self.dt = dt
        self.m = self.λ[0]
        self.k = self.λ[1]
        self.b = self.λ[2]
        self.update_matrices()
        
    def update_matrices(self):
        # Recalculate the state transition matrix Φ and the input transition matrix B
        self.Φ = array([[1, self.dt],
                        [- (self.k * self.dt) / self.m, 1 - ((self.b / self.m) * self.dt)]])

        self.B = array([[0],
                        [self.dt/self.m]])


def ensure_positive_semidefinite(matrix: ndarray) -> ndarray:
    """Ensure that a matrix is positive semidefinite."""
    symmetric_matrix = (matrix + matrix.T) / 2
    eigvals = linalg.eigvalsh(symmetric_matrix)
    min_eigval = min(eigvals)
    if min_eigval < 0:
        symmetric_matrix += eye(symmetric_matrix.shape[0]) * (-min_eigval + 1e-8)
    return symmetric_matrix

if __name__ == "__main__":
    seed(42)
    num_simulations = 1000
    dt = 0.1
    num_steps = 100
    λ_values = uniform(10.0, 50.0, num_simulations)
    k_values = uniform(1.0, 5.0, num_simulations)
    b_values = uniform(1.0, 5.0, num_simulations)
    H = array([[1, 0]])  # Fixed H
    Q_values = [ensure_positive_semidefinite(eye(H.shape[1]) * uniform(0.01, 1.0)) for _ in range(num_simulations)]
    R_values = [ensure_positive_semidefinite(eye(H.shape[0]) * uniform(0.01, 1.0)) for _ in range(num_simulations)]

    for i in range(num_simulations):
        λ = λ_values[i]
        k = k_values[i]
        b = b_values[i]
        Q = Q_values[i]
        R = R_values[i]
        x0 = array([0.0, 0.0]).reshape(2, 1)    # Initial state
        u = array([5.0]).reshape(1, 1)          # Input

        # Initialize SimpleHarmonicOscillator
        sho = SimpleHarmonicOscillator(λ, k, b, dt, H, Q, R)

        # Assert initial values
        assert sho.λ == λ, f"Mass initialization mismatch: expected {λ}, got {sho.λ}"
        assert allclose(sho.Φ, array([[1, dt], 
                                      [- (k * dt) / λ, 1 - ((b / λ) * dt)]])), "State transition matrix mismatch"
        assert allclose(sho.B, array([[0], 
                                      [dt / λ]])), "Input transition matrix mismatch"
        assert allclose(sho.H, H), "Measurement matrix mismatch"
        assert allclose(sho.Q, Q), "Process noise covariance matrix mismatch"
        assert allclose(sho.R, R), "Measurement noise covariance matrix mismatch"
        print(f"Test {i+1}/{num_simulations}: Initialization checks passed.")

        # Test parent class methods
        z, v = sho.output(x0, noisy=True)
        assert z.shape == (1, 1), "Output shape mismatch"
        print(f"Test {i+1}/{num_simulations}: Output method check passed.")
        
        x_next, w = sho.dynamics(x0, u, noisy=True)
        assert x_next.shape == (2, 1), "Dynamics state shape mismatch"
        print(f"Test {i+1}/{num_simulations}: Dynamics method check passed.")

        x_updated, z_updated, w_updated, v_updated = sho.update(x0, u, noisy=True)
        assert x_updated.shape == (2, 1), "Update state shape mismatch"
        assert z_updated.shape == (1, 1), "Update output shape mismatch"
        print(f"Test {i+1}/{num_simulations}: Update method check passed.")

        # Test consistency over multiple steps
        states = [x0.flatten()]
        for step in range(num_steps):
            x_updated, _, _, _ = sho.update(x_updated, u, noisy=True)
            states.append(x_updated.flatten())
        
        states = array(states)
        assert states.shape == (num_steps + 1, 2), "States array shape mismatch"
        print(f"Test {i+1}/{num_simulations}: Multiple steps consistency check passed.")


    print(f"SimpleHarmonicOscillator class Monte Carlo tests ({num_simulations} simulations) passed.")


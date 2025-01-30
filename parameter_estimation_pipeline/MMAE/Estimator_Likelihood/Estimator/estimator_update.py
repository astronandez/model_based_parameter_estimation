import numpy as np
from numpy import ndarray, linalg, eye

from System.Model.system import System
from .estimator_history import EstimatorHistory

# For Testing
from System.system_simulator import SystemSimulator

''' State_Estimator:

The state_estimator class was created based on the Standard Kalman Filter implementation
where the following functions were used (@ signifies dot product):

'''
class EstimatorUpdate:
    def __init__(self, s: System, P0: ndarray, x0: ndarray):
        self.s = s
        self._P = P0
        self.x0 = x0
        self.h = EstimatorHistory([self.x0], [self._P], [], [])
    
    
    # Only two inputs are allowed for state estimation u, and z and only update should be called
    # externally. Consolidated the predict and update steps into one function
    def update(self, z: ndarray, _x̂: ndarray, ẑ: ndarray) -> 'tuple[ndarray, ndarray]':
        ''' Predict Phase of Kalman Filter
        
            x[t] = A @ x[t-1] + B @ u[t-1]              (1) Computed within the plant
            P[t] = A @ P[t-1] @ A.T + Q                 (2)
            
        '''
        _P = self.s.Φ @ self._P @ self.s.Φ.T + self.s.Q
        ''' Update Phase of Kalman Filter
        
            K = P[t] @ H.T @ (H @ P[t] @ H.T + R)^-1    (4)
            x[t] = x[t] + K @ (z[t] - H @ x[t])         (5)
            P[t] = P[t] - K @ H @ P[t]                  (6)

        '''
        A = ((self.s.H @ _P @ self.s.H.T) + self.s.R)   # Residual covariance
        
        K = _P @ self.s.H.T @ linalg.inv(A)             # Kalman Gain
        r = z - ẑ                                       # Residual (innovation)
        x̂ = _x̂ + K @ (r)                                # Corrected state prediction
        P = _P - K @ self.s.H @ _P                      # Corrected prediction covariance 
        
        self._x, self._P = x̂, P                         # Update previous state
        self.h.append(x̂, P, _x̂, _P)                     # Track history
        
        return x̂, P, r, A
    

########### Testbench ###########

# Ensure positive semidefinite matrix
def ensure_positive_semidefinite(matrix: np.ndarray) -> np.ndarray:
    symmetric_matrix = (matrix + matrix.T) / 2
    eigvals = np.linalg.eigvalsh(symmetric_matrix)
    min_eigval = min(eigvals)
    if min_eigval < 0:
        symmetric_matrix += eye(symmetric_matrix.shape[0]) * (-min_eigval + 1e-8)
    return symmetric_matrix

# Run simulations
def run_estimator_simulation(Q_scale, R_scale, num_simulations=1000, num_steps=100, dt=0.1):
    np.random.seed(42)  # For reproducibility
    λ_values = np.random.uniform(10.0, 50.0, num_simulations)
    k_values = np.random.uniform(1.0, 5.0, num_simulations)
    b_values = np.random.uniform(1.0, 5.0, num_simulations)
    H = np.array([[1, 0]])  # Fixed H
    Q_values = [ensure_positive_semidefinite(np.eye(H.shape[1]) * Q_scale * np.random.uniform(0.01, 1.0)) for _ in range(num_simulations)]
    R_values = [ensure_positive_semidefinite(np.eye(H.shape[0]) * R_scale * np.random.uniform(0.01, 1.0)) for _ in range(num_simulations)]

    for i in range(num_simulations):
        λ = λ_values[i]
        k = k_values[i]
        b = b_values[i]
        Q = Q_values[i]
        R = R_values[i]
        x0 = np.array([0.0, 0.0]).reshape(2, 1)  # Initial state
        u = np.array([5.0]).reshape(1, 1)  # Control input

        # Initialize SystemSimulator for the actual plant (noisy)
        plant_simulator = SystemSimulator(λ, k, b, dt, H, Q, R, x0, noisy=True)
        P0 = np.eye(plant_simulator.model.Φ.shape[1])

        # Initialize EstimatorUpdate
        estimator_update = EstimatorUpdate(plant_simulator.model, P0, x0)

        for t in range(num_steps):
            x_true, z = plant_simulator.update(u)

            # Use uniformly distributed random values for ẑ and _x̂
            dummy_ẑ = np.random.uniform(-10, 10, (1, 1))  # Dummy predicted measurement
            dummy_x̂ = np.random.uniform(-10, 10, (2, 1))  # Dummy predicted state

            # Perform state estimation
            x̂, P, r, A = estimator_update.update(z, dummy_x̂, dummy_ẑ)

            # Validate results using assert statements
            assert x̂.shape == (2, 1), "State estimate shape mismatch"
            assert P.shape == (2, 2), "Covariance matrix shape mismatch"
            assert r.shape == (1, 1), "Residual shape mismatch"
            assert A.shape == (1, 1), "Covariance shape mismatch"
            print(f"Iteration {i+1}, Step {t+1}: Shape checks passed.")

            # Check Kalman Gain K
            P_pred = estimator_update.s.Φ @ P0 @ estimator_update.s.Φ.T + estimator_update.s.Q  # Predicted covariance
            K = P_pred @ estimator_update.s.H.T @ linalg.inv(estimator_update.s.H @ P_pred @ estimator_update.s.H.T + estimator_update.s.R)
            # assert np.allclose(K, P @ estimator_update.s.H.T @ linalg.inv(A)), "Kalman Gain K computation mismatch"
            # print(f"Iteration {i+1}, Step {t+1}: Kalman Gain K check passed.")

            # Check updated state estimate x̂
            expected_x̂ = dummy_x̂ + K @ (r)
            assert np.allclose(x̂, expected_x̂), "Updated state estimate x̂ computation mismatch"
            print(f"Iteration {i+1}, Step {t+1}: Updated state estimate x̂ check passed.")

            # Check updated covariance matrix P
            expected_P = P_pred - K @ estimator_update.s.H @ P_pred
            assert np.allclose(P, expected_P), "Updated covariance matrix P computation mismatch"
            print(f"Iteration {i+1}, Step {t+1}: Updated covariance matrix P check passed.")

            # Check residual r
            expected_r = z - dummy_ẑ
            assert np.allclose(r, expected_r), "Residual r computation mismatch"
            print(f"Iteration {i+1}, Step {t+1}: Residual r check passed.")

            # Check covariance matrix A
            expected_A = estimator_update.s.H @ P_pred @ estimator_update.s.H.T + estimator_update.s.R
            assert np.allclose(A, expected_A), "Covariance matrix A computation mismatch"
            print(f"Iteration {i+1}, Step {t+1}: Covariance matrix A check passed.")

            # Update P0 for the next step
            P0 = P

        print(f"Simulation {i+1}/{num_simulations} completed.")

    print(f"EstimatorUpdate class Monte Carlo tests ({num_simulations} simulations) completed for Q_scale={Q_scale} and R_scale={R_scale}.")

if __name__ == "__main__":
    num_simulations = 1000
    num_steps = 100
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

    for title_suffix, Q_scale, R_scale in combinations:
        run_estimator_simulation(Q_scale, R_scale, num_simulations, num_steps, dt)
    
    print(f"EstimatorUpdate class Monte Carlo tests ({num_simulations} simulations) completed for all Q and R combinations.")
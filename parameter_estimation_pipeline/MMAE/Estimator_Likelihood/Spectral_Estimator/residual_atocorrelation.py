import numpy as np

import numpy as np

class ResidualAutocorrelation:
    def __init__(self, N: int, d: int):
        """
        Efficient residual autocorrelation estimator.

        Parameters:
        - N: Number of past residuals to store for computing autocorrelation.
        - d: Dimension of residual vector r_k.
        """
        self.N = N  # Window size
        self.d = d  # Dimension of residual vector
        self.residual_history = np.zeros((N, d))  # Circular buffer
        self.index = 0  # Circular buffer index
        self.filled = False  # Track if buffer is full

    def update(self, r_k: np.ndarray):
        """
        Update the residual history and compute autocorrelation.

        Parameters:
        - r_k: Current residual vector for model k.

        Returns:
        - A_k: Autocorrelation matrix estimate (Equation 18).
        """
        # Ensure residual has correct shape (d,)
        r_k = r_k.flatten()

        # Update the circular buffer
        self.residual_history[self.index] = r_k
        self.index = (self.index + 1) % self.N
        if self.index == 0:
            self.filled = True  # Buffer is full

        # Number of valid samples in buffer
        N_eff = self.N if self.filled else self.index
        if N_eff < 2:
            return np.zeros((self.d, self.d))  # Return zero matrix if not enough data

        # Compute unbiased autocorrelation function
        A_k = np.zeros((self.d, self.d))
        for p in range(N_eff):
            residuals_shifted = self.residual_history[:N_eff - p]
            residuals_delayed = self.residual_history[p:N_eff]
            A_k += np.dot(residuals_shifted.T, residuals_delayed) / (N_eff - p)

        A_k /= N_eff  # Normalize by number of samples
        return A_k

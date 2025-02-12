import numpy as np

from .residual_atocorrelation import ResidualAutocorrelation
from .power_spectral_density import PowerSpectralDensity

from collections import deque  # Efficient rolling window

class SpectralEstimator:
    def __init__(self, N, d, psd_window_size=10):
        """
        Spectral Estimator with PSD computation and rolling history.

        Parameters:
        - N: Number of samples for autocorrelation estimation.
        - d: Residual vector dimension.
        - psd_window_size: Number of past PSD values to store for expected PSD computation.
        """
        self.ResidualAutocorr = ResidualAutocorrelation(N, d)  # Autocorrelation computed over N samples
        self.PowerSpectralDensity = PowerSpectralDensity(N, d)  # Uses same N for frequency resolution
        self.psd_history = deque(maxlen=psd_window_size)  # Rolling buffer for PSD expectation

    def update(self, r_k):
        """
        Updates the estimator with a new residual and computes the power spectral density.

        Parameters:
        - r_k: Residual vector at current time step.

        Returns:
        - r_k_gamma: Spectral residual at frequency f0.
        - A_k_gamma: Covariance of the spectral residual.
        """
        # Compute autocorrelation using N samples
        A_k = self.ResidualAutocorr.update(r_k)

        # Compute power spectral density
        PSD = self.PowerSpectralDensity.update(A_k)

        # Store in history buffer
        self.psd_history.append(PSD)

        # Compute expectation of PSD using a rolling average
        PSD_mean = np.mean(self.psd_history, axis=0) if len(self.psd_history) > 1 else PSD

        # Compute spectral residual
        r_k_gamma = PSD - PSD_mean  # Deviation from expected PSD
        r_k_gamma = (PSD - PSD_mean).reshape(-1, 1)  # Ensure column vector

        # Compute covariance of spectral residuals
        if len(self.psd_history) > 1:
            A_k_gamma = np.cov(np.array(self.psd_history).reshape(-1, PSD.size).T)
        else:
            A_k_gamma = np.eye(PSD.shape[0]) * 1e-6  # Small regularization for stability
        
        A_k_gamma = A_k_gamma.reshape(-1, 1)
        
        return r_k_gamma, A_k_gamma
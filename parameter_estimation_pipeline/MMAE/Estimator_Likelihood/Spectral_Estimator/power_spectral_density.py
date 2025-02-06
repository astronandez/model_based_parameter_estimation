import numpy as np

import numpy as np

class PowerSpectralDensity:
    def __init__(self, N: int, d: int):
        self.N = N  # Window size
        self.d = d  # Residual vector dimension

    def update(self, A_k: np.ndarray):
        """
        Compute the Power Spectral Density (PSD) from autocorrelation A_k(p).

        Parameters:
        - A_k: Autocorrelation matrix for estimator k.

        Returns:
        - PSD: Power spectral density matrix.
        """
        if self.d == 1:
            # Scalar case (Equation 20): Apply FFT to time-domain residuals
            PSD = np.fft.fft(A_k.flatten(), axis=0)
            PSD = (1 / self.N) * np.abs(PSD) ** 2
        else:
            # Vector case (Equation 19): Use FFT on autocorrelation function
            PSD = np.zeros((self.d, self.d), dtype=complex)
            for p in range(-self.N + 1, self.N):
                PSD += A_k * np.exp(-1j * 2 * np.pi * p / self.N)
            PSD = np.abs(PSD)  # Take magnitude

        return PSD

from numpy import ndarray, zeros, atleast_1d, atleast_2d, reshape, arange, array
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal

def noise(cov: ndarray, noisy: bool = True) -> ndarray:
    cov = atleast_2d(cov)
    dim = cov.shape[0]
    
    if noisy:
        return multivariate_normal(zeros(dim), cov)
    else:
        return zeros(dim)

class System:
    def __init__(self, Φ: ndarray, B: ndarray, H: ndarray, Q: ndarray, R: ndarray, λ=None):
        self.Φ = Φ
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R

        self.λ = λ

    def output(self, x: ndarray, noisy: bool) -> 'tuple[ndarray, ndarray]':
        v = noise(self.R, noisy)
        v = reshape(v, (v.shape[0], 1))
        z = self.H @ x + v
        return z, v

    def dynamics(self, x: ndarray, u: ndarray, noisy: bool) -> 'tuple[ndarray, ndarray]':
        x, u = atleast_1d(x), atleast_1d(u)
        w = noise(self.Q, noisy)
        w = reshape(w, (w.shape[0], 1))
        x = self.Φ @ x + self.B @ u + w
        return x, w

    def update(self, x: ndarray, u: ndarray, noisy: bool) -> 'tuple[ndarray, ndarray, ndarray, ndarray]':
        x, u = atleast_1d(x), atleast_1d(u)
        x, w = self.dynamics(x, u, noisy)
        z, v = self.output(x, noisy)

        # General assertions to confirm shapes are correct
        assert x.shape == (self.Φ.shape[0], 1), f"State shape mismatch: expected {(self.Φ.shape[0], 1)}, got {x.shape}"
        assert w.shape == (self.Φ.shape[0], 1), f"Process noise shape mismatch: expected {(self.Φ.shape[0], 1)}, got {w.shape}"
        assert z.shape == (self.H.shape[0], 1), f"Output shape mismatch: expected {(self.H.shape[0], 1)}, got {z.shape}"
        assert v.shape == (self.H.shape[0], 1), f"Measurement noise shape mismatch: expected {(self.H.shape[0], 1)}, got {v.shape}"


        return x, z, w, v



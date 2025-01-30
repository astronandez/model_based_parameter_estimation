from dataclasses import dataclass
from numpy import ndarray

# Estimator_History maintains track of the stored state estimator predictions and covariances when 
# incrementing the timestep t
@dataclass
class EstimatorHistory:
    x: list[ndarray]
    P: list[ndarray]
    _x: list[ndarray]
    _P: list[ndarray]

    def append(self, x: ndarray, P: ndarray, _x: ndarray, _P: ndarray):
        self.x.append(x)
        self.P.append(P)
        self._x.append(_x)
        self._P.append(_P)
        
    def output_history(self) -> tuple[list[ndarray], list[ndarray], list[ndarray], list[ndarray]]:
        return self.x, self.P, self._x, self._P
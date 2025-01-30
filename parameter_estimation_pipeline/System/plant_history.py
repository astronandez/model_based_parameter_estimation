from dataclasses import dataclass
from numpy import ndarray

# Plant_History maintains track of the stored plant datatraces when incrementing the timestep t
@dataclass
class PlantHistory:
    x: list[ndarray]
    u: list[ndarray]
    z: list[ndarray]
    w: list[ndarray]
    v: list[ndarray]

    def append(self, x: ndarray, u: ndarray, z: ndarray, w: ndarray, v: ndarray):
        self.x.append(x)
        self.u.append(u)
        self.z.append(z)
        self.w.append(w)
        self.v.append(v)
    
    def output_history(self) -> tuple[list[ndarray], list[ndarray], list[ndarray], list[ndarray], list[ndarray]]:
        return self.x, self.u, self.z, self.w, self.v

    

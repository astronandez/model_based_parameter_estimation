from numpy import ndarray, linspace, repeat, zeros, zeros_like, array, reshape, arange
from numpy.random import normal
from numpy import sin, pi
from numpy import heaviside, piecewise
from numpy.random import randn

from System.Model.system import System

# base class input, derived classes
class Input:
    def __init__(self, s: System, T: float, u0: ndarray = None):
        """
        Initializes an input class object that can generate various input signals.
        
        :param s: The System that provides information on the input dimensions.
        :param T: The period of generated input signals.
        :param u0: The initial size and value of input vector u.
        """
        self.s = s
        self.T = T
        self.u0 = zeros((self.s.B.shape[-1], 1)) if u0 is None else u0

    def step_function(self, steps, amplitude, change=100):
        """
        Generate a step function input signal.
        
        :param steps: The total number of time steps.
        :param amplitude: The amplitude of the step.
        :return: A numpy array representing the step function input signal.
        """
        u = zeros((steps, self.s.B.shape[1]))
        u[change:] = amplitude
        return u

import numpy as np


class tanh:
    # Hyperbolic tanh ActFunctions f(x) = tannh(ax)
    def __init__(self):
        self.expo = 1
        self.range = [-1, 1]

    def __str__(self):
        return "Function: tanh(x)"

    def setExponent(self, value=2):
        self.expo = value

    def evaluate(self, value):
        return np.tan(self.expo * value)
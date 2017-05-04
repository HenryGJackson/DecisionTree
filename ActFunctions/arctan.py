import numpy as np



class Arctan:
    # Inverse tan ActFunctions f(x) = arctan(ax)
    def __init__(self):
        self.coeff = 1
        self.range = [-0.5 * np.pi, 0.5 * np.pi]

    def __str__(self):
        return "Function: arctan(x)"

    def setParam(self, value):
        self.coeff = value

    def evaluate(self, value):
        return np.arctan(self.coeff * value)
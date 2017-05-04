import numpy as np


class Gaussian:
    def __init__(self):
        self.range = [0, 1]

    def __str__(self):
        return "Function: Gaussian"

    def evaluate(self, x):
        return np.exp((-1) * x * x)
        # End Individual Function Classes
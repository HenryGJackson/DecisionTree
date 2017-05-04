import numpy as np


class sinusoid:
    def __init__(self):
        self.range = [-1, 1]

    def __str__(self):
        return "Function: Sinusoid"

    def evaluate(self, x):
        return np.sin(x)
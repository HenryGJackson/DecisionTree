import numpy as np


class soft_step:
    # Soft step ActFunctions f(x) = 1 / (1 + exp(-ax))
    def __init__(self):
        self.expo = 1
        self.range = [0, 1]

    def __str__(self):
        return "Function: Soft Step Function"

    def setExponent(self, value):
        self.expo = value

    def evaluate(self, value):
        return 1.0 / (1 + np.exp(-self.expo * value))
import numpy as np


class soft_plus:
    # f(x) = ln ( 1 + exp(x) )
    def __init__(self):
        self.range = [0, float("inf")]

    def __str__(self):
        return "Function: Soft Plus"

    def evaluate(self, x):
        return np.log(1 + np.exp(x))


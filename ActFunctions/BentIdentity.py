import numpy as np


class BentIdentity:
    # f(x) = 0.5 * (sqrt(x^2 + 1) - 1) + x
    def __init__(self):
        self.range = [-float("inf"), float("inf")]

    def __str__(self):
        return "Function: Bent Identity"

    def evaluate(self, x):
        return 0.5 * (np.sqrt(x * x + 1) - 1) + x

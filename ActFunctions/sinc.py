import numpy as np


class sinc:
    def __str__(self):
        return "Function: Sinc"

    def evaluate(self, x):
        if x == 0:
            return 1
        else:
            return np.sin(x) / x

class PReLU:
    # parametric rectified Linear Unit f(x) = ax, x < 0
    #                           f(x) = bx, x >= 0
    def __init__(self):
        self.coeff = 1
        self.coeff2 = 1
        self.range = [-float("inf"), float("inf")]

    def __str__(self):
        return "Function: PReLU"

    def setParam(self, value, value2):
        self.coeff = value
        self.coeff2 = value2

    def evaluate(self, value):
        if value < 0:
            return self.coeff2 * value
        else:
            return self.coeff * value

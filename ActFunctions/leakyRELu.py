class LeakyReLU:
    # Leaky ectified Linear Unit f(x) = 0.01ax, x < 0
    #                           f(x) = ax,     x >= 0
    def __init__(self):
        self.coeff = 1
        self.coeff2 = 1
        self.range = [-float("inf"), float("inf")]

    def __str__(self):
        return "Function: leakyReLU"

    def setParam(self, value, value2):
            self.coeff = value
            self.coeff2 = value2

    def evaluate(self, value):
        if value < 0:
            return 0.01 * self.coeff2 * value
        else:
            return self.coeff * value
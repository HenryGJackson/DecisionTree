class ReLU(function):
    # Rectified Linear Unit f(x) = 0, x < 0
    #                      f(x) = ax, x >= 0
    def __init__(self):
        self.coeff = 1
        self.range = [0, float("inf")]

    def __str__(self):
        return "Function: ReLU"

    def setParam(self, value):
        self.coeff = value

    def evaluate(self, value):
        if value < 0:
            return 0
        else:
            return self.coeff * value













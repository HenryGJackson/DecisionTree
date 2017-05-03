class ELU:
    # Exponential linear unit
    def __init__(self, value):
        self.coeff = value
        self.range = [-value, float("inf")]

    def __str__(self):
        return "Function: ELU"

    def setParam(self, value):
        self.coeff = value

    def evaluate(self, value):
        if value < 0:
            return self.coeff * (exp(value) - 1)
        else:
            return value
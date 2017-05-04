class softsign:
    # Soft sign ActFunctions f(x) = x / (1 + |x|)
    def __init__(self):
        self.coeff = 1
        self.range = [-1, 1]

    def __str__(self):
        return "Function: Soft Sign Function"

    def setParam(self, value):
        self.coeff = value

    def evaluate(self, value):
        if value < 0:
            return value / (1 - value)
        else:
            return value / (1 + value)
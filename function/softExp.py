class soft_exponential(function):
    def __init__(self, a):
        self.a = a
        self.range = [-float("inf"), float("inf")]
        return

    def __str__(self):
        return "Function: Soft Exponential"

    def evaluate(self, x):
        if self.a < 0:
            return (-1) * (np.log(1 - self.a * (x + self.a)))
        elif self.a == 0:
            return x
        else:
            return self.a + (np.exp(self.a * x) - 1) / self.a

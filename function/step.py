class step(function):
    # Step Function f(x) = 0, x < 0
    #              f(x) = 1, x >= 0
    def __init__(self):
        self.step = 0
        self.range = [0, 1]
        return

    def __str__(self):
        return "Function: Step Function"

    def setStep(self, value):
        self.step = value
        return

    def evaluate(self, val):
        if val < self.step:
            return 0
        else:
            return 1

    def tweak_param(self, up):
        if up:
            self.step += 1
        else:
            self.step -= 1
        return
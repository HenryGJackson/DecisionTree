class SReLU:
    # S-shaped rectified linear activation unit
    def __init__(self):
        self.tl = 0
        self.tr = 1
        self.al = 1
        self.ar = 1
        self.range = [-float("inf"), float("inf")]

    def __str__(self):
        return "Function: SReLU"

    def setParam(self, tl, tr, al, ar):
        self.tl = tl
        self.tr = tr
        self.al = al
        self.ar = ar

    def evaluate(self, x):
        if x <= self.tl:
            return self.tl + self.al * (x - self.tl)
        elif x < self.tr:
            return x
        else:
            return self.tr + self.ar * (x - self.tr)

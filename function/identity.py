# Function Member classes. Each function instance can only...
# ... invoke an instance of one of these classes
# Identity function f(x) = x
class identity:
    def __init__(self):
        self.value = 1
        self.range = [-float("inf"), float("inf")]

    def __str__(self):
        return "Function: Identity"

    def setValue(self, value):
        self.value = value

    def evaluate(self, val=0):
        return self.value * val

    def tweak_param(self, up):
        if up:
            self.value = self.value * 1.01
        else:
            self.value = self.value * 0.99
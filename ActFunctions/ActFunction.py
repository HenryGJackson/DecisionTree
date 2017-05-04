from ActFunctions.arctan import Arctan
from ActFunctions.APL import APL
from ActFunctions.BentIdentity import BentIdentity
from ActFunctions.ELU import ELU
from ActFunctions.Gaussian import Gaussian
from ActFunctions.PRelU import PReLU
from ActFunctions.ReLU import ReLU
from ActFunctions.SReLU import SReLU
from ActFunctions.identity import identity
from ActFunctions.leakyRELu import LeakyReLU
from ActFunctions.sinc import sinc
from ActFunctions.sine import sinusoid
from ActFunctions.softExp import soft_exponential
from ActFunctions.softPlus import soft_plus
from ActFunctions.softStep import soft_step
from ActFunctions.softsign import softsign
from ActFunctions.step import step
from ActFunctions.tanh import tanh

def get_larger(val1, val2):
    if val1 > val2:
        return val1
    else:
        return val2

def array_length_check(pass_array, fail_array):
    if len(pass_array) < 1 | len(fail_array) < 1:
        print("Error: neuron->array_length_check: pass and fail training sets must have length > 0")
        if len(pass_array) < 1:
            print("Error source --> pass set")
        if len(fail_array) < 1:
            print("Error source --> fail set")
        return False
    elif len(pass_array) != len(fail_array):
        print("Error: neuron->array_length_check: Both the training arrays must have equal length")
        return False
    else:
        return True


class ActFunction:
    # Initialisation methods: class: ActFunctions
    def __init__(self):
        self.failind = 1
        self.passind = 0
        self.func_num = None
        self.func = None
        self.func_lock = False
        self.cut = None
        self.cutdir = True
        self.maxi = -1
        self.mini = 1
        return
        # End: Initialisation methods: class: ActFunctions

    def change_function(self, funct_num):
        if not self.func_lock:
            if funct_num == 0:
                self.func = identity()
            elif funct_num == 1:
                self.func = step()
            elif funct_num == 2:
                self.func = soft_step()
            elif funct_num == 3:
                self.func = tanh()
            elif funct_num == 4:
                self.func = Arctan()
            elif funct_num == 5:
                self.func = softsign()
            elif funct_num == 6:
                self.func = ReLU()
            elif funct_num == 7:
                self.func = LeakyReLU()
            elif funct_num == 8:
                self.func = PReLU()
            elif funct_num == 9:
                self.func = ELU(1)
            elif funct_num == 10:
                self.func = SReLU()
            elif funct_num == 17:
                self.func = APL()
            elif funct_num == 12:
                self.func = soft_plus()
            elif funct_num == 13:
                self.func = BentIdentity()
            elif funct_num == 14:
                self.func = soft_exponential(1)
            elif funct_num == 15:
                self.func = sinusoid()
            elif funct_num == 16:
                self.func = sinc()
            elif funct_num == 11:
                self.func = Gaussian()
            else:
                raise Exception("ActFunctions number invalid, range = [0,17]")
        else:
            raise Exception("Tried to change function after lock")

    def __str__(self):
        return self.func.__str__()
        # Function Evaluation methods

    def set_cut(self, cut, greater_than=True):
        self.cut = cut
        self.cutdir = greater_than
        return

    def get_cut(self):
        return self.cut

    def pass_cut(self, value):
        if self.cutdir & (value > self.cut):
            return True
        elif (self.cutdir == False) & (value < self.cut):
            return True
        else:
            return False

    def evaluate(self, x):
        return self.func.evaluate(x)

    def cut_inverted(self):
        return self.cutdir


    # Returns two arrays of values after the initial values have been passed through the activation ActFunctions
    def apply_function(self, pass_array, fail_array):
        # if array_length_check(pass_array, fail_array):
        pass_results = []
        fail_results = []
        iters = get_larger(len(pass_array),len(fail_array))
        # print (pass_array)
        for i in range(iters):
            # print("apply_function",i,pass_array[i])
            if i < len(pass_array):
                pass_results.append(self.func.evaluate(pass_array[i]))
            if i < len(fail_array):
                fail_results.append(self.func.evaluate(fail_array[i]))
            # print("loop\n\n")
        return pass_results, fail_results
        # else:
        #     raise Exception("Array Length Error")

    def get_function_range(self, pass_output, fail_output):
        # if array_length_check(pass_output, fail_output):
            # pass_output = pass_output.reset_index()
            # fail_output = fail_output.reset_index()
            # print (len(pass_output))
        self.mini = pass_output[0]
        self.maxi = pass_output[0]
        iters = get_larger(len(pass_output), len(fail_output))
        for i in range(iters):
            if i < len(pass_output):
                if pass_output[i] < self.mini:
                    self.mini = pass_output[i]
                elif pass_output[i] > self.maxi:
                    self.maxi = pass_output[i]
            if i < len(fail_output):
                if fail_output[i] < self.mini:
                    self.mini = fail_output[i]
                if fail_output[i] > self.maxi:
                    self.maxi = fail_output[i]

        # else:
        #     raise Exception("Array Length Error")

    # Finds the best cut to use for a ActFunctions which is locked in.
    # Iterates over the range of output for the ActFunctions with number of iterations = step_number
    # Maximises the ratio of correctly identified points to incorrectly identified
    # Returns the best cut value and the ratio of pass/fail (including both sets)
    # Means that to find the best method only requires comparing output pf this ActFunctions for each method
    def find_best_cut(self, pass_array, fail_array, step_number=10):
        # if array_length_check(pass_array, fail_array):
        func_output = self.apply_function(pass_array, fail_array)
        # print (func_output)
        self.get_function_range(func_output[self.passind], func_output[self.failind])
        step_size = float(self.maxi - self.mini / step_number)
        best_ratio = 0
        best_cut = 0
        iters = get_larger(len(pass_array), len(fail_array))
        for i in range(step_number):
            cut_val = i * step_size
            self.set_cut(cut_val)
            correct = 0
            incorrect = 0
            for j in range(iters):
                if j < len(pass_array):
                    if self.pass_cut(pass_array[j]):
                        correct += 1
                    else:
                        incorrect += 1
                if j < len(fail_array):
                    if self.pass_cut(fail_array[j]):
                        incorrect += 1
                    else:
                        correct += 1
            ratio = float(correct / (correct + incorrect))

            if ratio > best_ratio:
                best_ratio = ratio
                best_cut = cut_val
                direct = True
            if (1 - ratio) > best_ratio:
                best_ratio = ratio
                best_cut = cut_val
                direct = False
                # print("find_best_cut, method: ", self.func.__str__(), ", Ratio: ", best_ratio)
        return best_cut, best_ratio, direct
        # else:
        #     raise Exception("Array Length Error")
            # End Trinings methods

            # Individual Function Classes

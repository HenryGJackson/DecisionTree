class function:
    # Initialisation methods: class: function
    def __init__(self, funct_num):
        self.func_num = funct_num
        if funct_num == 0:
            self.func = self.identity()
            self.func_lock = True
        elif funct_num == 1:
            self.func = self.step()
            self.func_lock = True
        elif funct_num == 2:
            self.func = self.soft_step()
            self.func_lock = True
        elif funct_num == 3:
            self.func = self.tanh()
            self.func_lock = True
        elif funct_num == 4:
            self.func = self.arctan()
            self.func_lock = True
        elif funct_num == 5:
            self.func = self.softsign()
            self.func_lock = True
        elif funct_num == 6:
            self.func = self.ReLU()
            self.func_lock = True
        elif funct_num == 7:
            self.func = self.leakyReLU()
            self.func_lock = True
        elif funct_num == 8:
            self.func = self.PReLU()
            self.func_lock = True
        elif funct_num == 9:
            self.func = self.ELU(1)
            self.func_lock = True
        elif funct_num == 10:
            self.func = self.SReLU()
            self.func_lock = True
        elif funct_num == 17:
            self.func = self.APL()
            self.func_lock = True
        elif funct_num == 12:
            self.func = self.soft_plus()
            self.func_lock = True
        elif funct_num == 13:
            self.func = self.bent_identity()
            self.func_lock = True
        elif funct_num == 14:
            self.func = self.soft_exponential(1)
            self.func_lock = True
        elif funct_num == 15:
            self.func = self.sinusoid()
            self.func_lock = True
        elif funct_num == 16:
            self.func = self.sinc()
            self.func_lock = True
        elif funct_num == 11:
            self.func = self.gaussian()
            self.func_lock = True
        else:
            print("function number invalid, range = [0,17]")
            self.func_lock = False
        return
        # End: Initialisation methods: class: function

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


        # End: Function Evaluation methods

        # Training methods

    # Returns two arrays of values after the initial values have been passed through the activation function
    def apply_function(self, pass_array, fail_array):
        if (array_length_check(pass_array, fail_array)):
            if self.func_lock:
                pass_results = []
                fail_results = []
                # print (pass_array)
                for i in range(len(pass_array)):
                    # print("apply_function",i,pass_array[i])
                    pass_results.append(self.func.evaluate(pass_array[i]))
                    fail_results.append(self.func.evaluate(fail_array[i]))
                    # print("loop\n\n")
                return (pass_results, fail_results)
            else:
                print("Error: neuron->function->apply_function: Function must be set and locked")
        else:
            print("neuron->function->apply_function: exiting...")
            return None

    def get_function_range(self, pass_output, fail_output):
        if (array_length_check(pass_output, fail_output)):
            # pass_output = pass_output.reset_index()
            # fail_output = fail_output.reset_index()
            # print (len(pass_output))
            mini = pass_output[0]
            maxi = pass_output[0]
            for i in range(len(pass_output)):
                if pass_output[i] < fail_output[i]:
                    if pass_output[i] < mini:
                        mini = pass_output[i]
                    if fail_output[i] > maxi:
                        maxi = fail_output[i]
                else:
                    if pass_output[i] > maxi:
                        maxi = pass_output[i]
                    if fail_output[i] < mini:
                        mini = fail_output[i]
            myrange = [mini, maxi]
            return myrange
        else:
            print("neuron->function->get_function_range: exiting...")
            return None

    def plot_hist(self, pass_array, fail_array):
        output = self.apply_function(pass_array, fail_array)
        plot_hist(output[0], output[1])
        return

    # Finds the best cut to use for a function which is locked in.
    # Iterates over the range of output for the function with number of iterations = step_number
    # Maximises the ratio of correctly identified points to incorrectly identified
    # Returns the best cut value and the ratio of pass/fail (including both sets)
    # Means that to find the best method only requires comparing output pf this function for each method
    def find_best_cut(self, pass_array, fail_array, step_number=10):
        if (array_length_check(pass_array, fail_array)):
            self.passind = 0
            self.failind = 1
            func_output = self.apply_function(pass_array, fail_array)
            # print (func_output)
            myrange = self.get_function_range(func_output[self.passind], func_output[self.failind])
            # print (myrange[0],myrange[1])
            step_size = float((myrange[1] - myrange[0]) / step_number)
            best_ratio = 0
            for i in range(step_number):
                cut_val = i * step_size
                self.set_cut(cut_val)
                correct = 0
                incorrect = 0
                for j in range(len(pass_array)):
                    if self.pass_cut(pass_array[j]):
                        correct += 1
                    else:
                        incorrect += 1
                    if self.pass_cut(fail_array[j]):
                        incorrect += 1
                    else:
                        correct += 1
                ratio = float(correct / (correct + incorrect))
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_cut = cut_val
            return (cut_val, ratio)
        else:
            print("neuron->function->find_best_cut: exiting...")
            return None
            # End Trinings methods

            # Individual Function Classes
















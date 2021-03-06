# Variable Decision Tree -- By Henry Jackson
# the n'th layer has 2^(n-1) nodes
# When probability of success < some value, make this an output node
# Also build in
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys


def plot_hist(array1, array2):
    plt.hist(array1)
    plt.show
    plt.hist(array2)
    plt.show


# Global methods:
# Check both arrays are of non-zero length
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


# End: Global methods

class network:
    def __init__(self, max_depth=4, node_count=0):
        self.current_layer = 0
        self.max_depth = max_depth
        self.node_count = 0
        self.used_variables = []
        return

    def add_to_used_variables(self, variable):
        if self.used_variables == []:
            self.used_variables.append(variable)
        else:
            for i in self.used_variables:
                if i == variable:
                    return
            else:
                self.used_variables.append(variable)
        return

    def check_variable_use(self, variable):
        if self.used_variables == []:
            return False
        else:
            for i in self.used_variables:
                if variable == i:
                    return True
            else:
                return False

    def increment_layer(self):
        self.current_layer += 1

    def print_node(self, node):
        print("Node: ", node.id, "Type: ", node.get_type())
        if node.func_locked:
            if node.get_function().cut_inverted():
                print(node.get_variable, " < ", node.get_function().get_cut())
            else:
                print(node.get_variable, " > ", node.get_function().get_cut())

    def visualise_tree(self):
        node = self.seed
        print("*** SEED ***")
        self.print_node(node)
        print("*** Layer 1 Fail ****")
        self.print_node(node.get_failure_node())
        print("*** Layer 1 Pass ****")
        self.print_node(node.get_success_node())
        print("*** Layer 2 Fail ****")
        # self.print_node(node.get_failure_node().get_failure_node())
        self.print_node(node.get_success_node().get_failure_node())
        print("*** Layer 2 Pass ****")
        self.print_node(node.get_success_node().get_success_node())
        # self.print_node(node.get_failure_node().get_success_node())

    def get_current_layer(self):
        return self.current_layer

    def check_layer(self):
        if self.current_layer > self.max_depth:
            return False
        else:
            return True

            # Network Creation Methods

    def create_network(self, win_df, lose_df):
        print("Creating First Node")
        output_nodes = self.create_first_node(win_df, lose_df)
        # print (self.seed.get_weight())
        # self.seed.get_function().plot_hist(win_df, lose_df)
        print("First Node Created")
        if output_nodes > 1:
            print("Output Reached after 1st node")
            return
        # Get the nodes that stem from the input node
        tmp = self.seed
        # print (tmp.get_weight())
        if tmp is None:
            raise Exception("network->create_network: Could not create seed. Exiting...")
        layer = self.seed.get_non_output_daughters()
        self.increment_layer()
        # While there are still new nodes to be created and we haven't reached...
        # ... the max depth, keep building the network
        print("New Nodes Created")

        while self.check_layer():
            tmp_new_nodes = []
            new_inter_nodes = 0
            # Loop through all of the newly created node objects
            for node in layer:
                if node.is_output():
                    continue
                # Create the nodes that stem from this one
                if node.has_parent:
                    if node == node.get_parent().get_success_node():
                        win_df = node.get_parent().accept_pass_df()
                        lose_df = node.get_parent().accept_fail_df()
                    else:
                        win_df = node.get_parent().reject_pass_df()
                        lose_df = node.get_parent().reject_fail_df()
                else:
                    print_node(node)
                    print("node has no parent")
                    continue
                node = self.prepare_node(win_df, lose_df, node)
                node.cut_dataset(win_df, lose_df)
                node = self.prepare_node(node.get_parent())
                new_inter_nodes += self.branch_node(node)
                if not node.get_success_node().is_output():
                    tmp_new_nodes = tmp_new_nodes.append(node.get_success_node())
                if not node.get_failure_node().is_output():
                    tmp_new_nodes = tmp_new_nodes.append(node.get_failure_node())
                    # If this node isn't an output node, add the two new nodes to a temporary list
            if tmp_new_nodes == []:
                raise Exception("No new nodes created. exiting..")
            else:
                layer = tmp_new_nodes
            self.increment_layer()
        return

    def create_next_layer(self, node):
        if node == None:
            return None
        else:
            if node.is_output():
                return None
            # Create the nodes that are children of this node
            # print (node.get_weight())

            new_layer = self.create_next_nodes(node)

            if not new_layer:
                return None
            # If the node is output there are no more nodes to add
            if node.is_output():
                return None
            # Set Up each node activation ActFunctions
            yes_node = self.prepare_node(node.accept_pass_df, node.accept_fail_df, node.get_success_node())
            if yes_node == None:
                node.set_type(-1)
            no_node = self.prepare_node(node.reject_pass_df, node.reject_fail_df, node.get_failure_node())
            if no_node == None:
                node.set_type(-1)

            # Cut the datasets on each node
            if (node.is_output()):
                yes_node.cut_dataset(node.accept_pass_df, node.accept_fail_df)
                no_node.cut_dataset(node.reject_pass_df, node.reject_fail_df)
            self.node_count += 2
            # return the newly created nodes
            return (yes_node, no_node)

    # returns the number of non-output nodes
    def branch_node(self, node):
        output_nodes = self.check_output(self.seed)
        if output_nodes < 1:
            self.node_count += 1
            self.seed.create_daughter(True, self.node_count)
            self.node_count += 1
            self.seed.create_daughter(False, self.node_count)
            return 2
        elif output_nodes == 1:
            self.node_count += 1
            if self.success_node_exists:
                self.seed.create_daughter(False, self.node_count)
            else:
                self.seed.create_daughter(True, self.node_count)
            return 1
        else:
            return 0

    def create_first_node(self, win_df, lose_df):
        # find the best classification ActFunctions for two arrays of the values of ONE variable...
        # ... which correspond to winners and losers
        self.node_count += 1
        self.seed = self.neuron(self.node_count)
        self.seed.set_type(0)
        self.seed = self.prepare_node(win_df, lose_df, self.seed)
        self.increment_layer()
        self.seed.cut_dataset(win_df, lose_df)
        new_inter_nodes = self.branch_node(self.seed)
        # Function for neuron n now locked in with cut value and weight(=best cut ratio)
        return new_inter_nodes

    def create_output_node(self, node, direction, return_value):
        if direction:
            self.node_count += 1
            new_node = neuron(self.node_count)
            new_node.set_type(-1, return_value)
            node.set_success_path(new_node)
            # node.set_failure_path(None)
        else:
            self.node_count += 1
            new_node = neuron(self.node_count)
            new_node.set_type(-1, return_value)
            node.set_failure_path(new_node)
            # node.set_success_path(None)
        return

    def check_output(self, node):
        nodes_created = 0
        self.print_node(node)
        if node.is_output():
            return True
        elif (node.accept_pass_df.empty()):
            self.create_output_node(node, True, False)
            nodes_created += 1
        elif (node.accept_fail_df.empty()):
            self.create_output_node(node, True, True)
            nodes_created += 1
        elif (node.reject_pass_df.empty()):
            self.create_output_node(node, False, False)
            nodes_created += 1
        elif node.reject_fail_df.empty():
            self.create_output_node(node, False, True)
            nodes_created += 1
        if nodes_created > 2:
            print("network->check_output: Node had no entries passed to it")
            self.print_node(node)
        return nodes_created

    def prepare_node(self, win_df, lose_df, node):
        if node == None:
            return None
        else:
            if node.is_output():
                return None
            print("*** NEW NODE --- LEVEL: ", self.get_current_layer(), " , ID: ", node.id, "***")
            if not (node.is_input()):
                print("... PARENT: ", node.get_parent().id)
            else:
                print("This is the Input Node")
            # print("Progress: |",flush = True)
            best_ratio = 0.0
            variables = list(win_df)
            usable_variable = False
            for variable in variables:  # Need to create the variables array
                if (variable == "Species") | (self.check_variable_use(variable)):
                    continue  # print "VARIABLE: ", variable
                usable_variable = True

                winners = win_df[variable]
                losers = lose_df[variable]
                # print "CheckPoint"
                result = node.find_best_activation_function(winners, losers, 50)
                if result is None:
                    continue
                # result[0] = best method index
                # result[1] = best cut value
                # result[2] = best cut ratio
                # result[3] = (Values greater than cut are winners) True OR False
                if result[2] > best_ratio:
                    best_results = result
                    var = variable
                    # do for every variable to find which variable and method make the first node
            # print best_results
            if not usable_variable:
                return None
            node.set_function(best_results[0], best_results[1], best_results[2], best_results[3])
            print("VARIABLE: ", var)
            self.add_to_used_variables(var)
            # print (result)
            # print (node.__str__())
            node.set_variable(var)
            node.set_type(self.get_current_layer())
            return node

    def total_weight_of_path(self, current_node):
        weight = 1.0
        # print (current_node.get_weight())
        while (1):
            # print ("blah",current_node.get_weight())
            weight = weight * current_node.get_weight()
            # print (weight)
            # print(current_node.get_type())
            if current_node.is_input():
                print("TOTAL WEIGHT OF PATH: ", weight)
                return weight
            else:
                current_node = current_node.get_parent()
                if current_node == None:
                    return weight

    def create_next_nodes(self, current_node):
        # Must build in checks to see if the next node should be an output
        # print (current_node.get_weight())
        # output when probability of success (of each node in this path multipled) reaches some satisfaction value
        if current_node == None:
            print("network->create_next_nodes: Node Passed in is None")
            return False
        else:

            if (self.total_weight_of_path(current_node)) < 0.9:
                if current_node.has_result_dfs():
                    if current_node.accept_fail_df.empty | current_node.reject_pass_df.empty | current_node.accept_pass_df.empty | current_node.reject_fail_df.empty:
                        current_node.set_type(-1)
                        return False
                    self.node_count += 1
                    current_node.set_success_path(self.neuron(self.node_count))
                    current_node.get_success_node().set_parent(current_node, self.get_current_layer())
                    self.node_count += 1
                    current_node.set_failure_path(self.neuron(self.node_count))
                    current_node.get_failure_node().set_parent(current_node, self.get_current_layer())
                else:
                    print("network->create_next_nodes: Printing node information")
                    self.print_node(current_node)

            else:
                # Set the current node as output
                current_node.set_type(-1)
                return False
        return True

    # End: Network Creation methods

            # End: Neuron: Training methods

        class function:
            # Initialisation methods: class: ActFunctions
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
                    print("ActFunctions number invalid, range = [0,17]")
                    self.func_lock = False
                return
                # End: Initialisation methods: class: ActFunctions

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

            # Returns two arrays of values after the initial values have been passed through the activation ActFunctions
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
                        print("Error: neuron->ActFunctions->apply_function: Function must be set and locked")
                else:
                    print("neuron->ActFunctions->apply_function: exiting...")
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
                    print("neuron->ActFunctions->get_function_range: exiting...")
                    return None

            def plot_hist(self, pass_array, fail_array):
                output = self.apply_function(pass_array, fail_array)
                plot_hist(output[0], output[1])
                return

            # Finds the best cut to use for a ActFunctions which is locked in.
            # Iterates over the range of output for the ActFunctions with number of iterations = step_number
            # Maximises the ratio of correctly identified points to incorrectly identified
            # Returns the best cut value and the ratio of pass/fail (including both sets)
            # Means that to find the best method only requires comparing output pf this ActFunctions for each method
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
                    print("neuron->ActFunctions->find_best_cut: exiting...")
                    return None
                    # End Trining methods

                    # Individual Function Classes

            # Function Member classes. Each ActFunctions instance can only...
            # ... invoke an instance of one of these classes
            # Identity ActFunctions f(x) = x
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

            class step:
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

            class soft_step:
                # Soft step ActFunctions f(x) = 1 / (1 + exp(-ax))
                def __init__(self):
                    self.expo = 1
                    self.range = [0, 1]

                def __str__(self):
                    return "Function: Soft Step Function"

                def setExponent(self, value):
                    self.expo = value

                def evaluate(self, value):
                    return 1.0 / (1 + np.exp(-self.expo * value))

            class tanh:
                # Hyperbolic tanh ActFunctions f(x) = tannh(ax)
                def __init__(self):
                    self.expo = 1
                    self.range = [-1, 1]

                def __str__(self):
                    return "Function: tanh(x)"

                def setExponent(self, value=2):
                    self.expo = value

                def evaluate(self, value):
                    return np.tan(self.expo * value)

            class arctan:
                # Inverse tan ActFunctions f(x) = arctan(ax)
                def __init__(self):
                    self.coeff = 1
                    self.range = [-0.5 * np.pi, 0.5 * np.pi]

                def __str__(self):
                    return "Function: arctan(x)"

                def setParam(self, value):
                    self.coeff = value

                def evaluate(self, value):
                    return np.arctan(self.coeff * value)

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

            class ReLU:
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

            class leakyReLU:
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

            class PReLU:
                # parametric rectified Linear Unit f(x) = ax, x < 0
                #                           f(x) = bx, x >= 0
                def __init__(self):
                    self.coeff = 1
                    self.coeff2 = 1
                    self.range = [-float("inf"), float("inf")]

                def __str__(self):
                    return "Function: PReLU"

                def setParam(self, value, value2):
                    self.coeff = value
                    self.coeff2 = value2

                def evaluate(self, value):
                    if value < 0:
                        return self.coeff2 * value
                    else:
                        return self.coeff * value

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

            class APL:
                # Adaptive piecewise linear
                def __init__(self):
                    self.max = 1
                    self.range = [-float("inf"), float("inf")]

                def __str__(self):
                    return "Function: APL"

                def evaluate(self, x):
                    return 0
                    # to be completed

            class soft_plus:
                # f(x) = ln ( 1 + exp(x) )
                def __init__(self):
                    self.range = [0, float("inf")]

                def __str__(self):
                    return "Function: Soft Plus"

                def evaluate(self, x):
                    return np.log(1 + np.exp(x))

            class bent_identity:
                # f(x) = 0.5 * (sqrt(x^2 + 1) - 1) + x
                def __init__(self):
                    self.range = [-float("inf"), float("inf")]

                def __str__(self):
                    return "Function: Bent Identity"

                def evaluate(self, x):
                    return 0.5 * (np.sqrt(x * x + 1) - 1) + x

            class soft_exponential:
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

            class sinusoid:
                def __init__(self):
                    self.range = [-1, 1]

                def __str__(self):
                    return "Function: Sinusoid"

                def evaluate(self, x):
                    return np.sin(x)

            class sinc:
                def __str__(self):
                    return "Function: Sinc"

                def evaluate(self, x):
                    if x == 0:
                        return 1
                    else:
                        return np.sin(x) / x

            class gaussian:
                def __init__(self):
                    self.range = [0, 1]

                def __str__(self):
                    return "Function: Gaussian"

                def evaluate(self, x):
                    return np.exp((-1) * x * x)
                    # End Individual Function Classes


def normalise_data(df):
    for i, j in enumerate(df):
        if j == "Species":
            # print ("String")
            continue
        # print (df[j])
        maxi = max(df[j])
        df[j] = df[j] / maxi
        # print (df[j])
        # break
    return df


df = pd.read_csv("Iris.csv")
df = normalise_data(df)
net = network(5)
nwin_df = df[df["Species"] == "Iris-setosa"].reset_index()
nlose_df = df[df["Species"] != "Iris-setosa"].reset_index()
nwin_df = nwin_df.drop("Species", axis=1)
nlose_df = nlose_df.drop("Species", axis=1)
print(nwin_df)
# exit()
if len(nwin_df) != len(nlose_df):
    # print("Cutting Dataframe",len(win_df), len(lose_df))
    if len(nwin_df) > len(nlose_df):
        nwin_df = nwin_df[:len(nlose_df)]
    else:
        nlose_df = nlose_df[:len(nwin_df)]
# print(win_df)
# print(lose_df)
print("Cut Dataframe", len(nwin_df), len(nlose_df))

net.create_network(nwin_df, nlose_df)
print("network created")
net.visualise_tree()
# test_df =
event = df.iloc[130].drop("Species")
print(event)
answer = net.seed.pass_on(df.iloc[130].drop("Species"))
exit()
# print(answer)

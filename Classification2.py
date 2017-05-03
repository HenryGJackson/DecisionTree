#Variable Decision Tree -- By Henry Jackson
# the n'th layer has 2^(n-1) nodes
# When probability of success < some value, make this an output node
# Also build in
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_hist(array1, array2):
    plt.hist(array1)
    plt.show
    plt.hist(array2)
    plt.show


#Global methods:
# Check both arrays are of non-zero length
def array_length_check(pass_array, fail_array):
    if len(pass_array) < 1 | len(fail_array) < 1:
        print ("Error: neuron->array_length_check: pass and fail training sets must have length > 0")
        if len(pass_array) < 1:
            print ("Error source --> pass set")
        if len(fail_array) < 1:
            print ("Error source --> fail set")
        return False
    elif len(pass_array) != len(fail_array):
        print ("Error: neuron->array_length_check: Both the training arrays must have equal length")
        return False
    else:
        return True
#End: Global methods

class network:
    def __init__(self, max_depth = 4, node_count = 0):
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
        print ("Node: ", node.id, "Type: ", node.get_type())
        if node.func_locked:
            if node.get_function().cut_inverted():
                print (node.get_variable, " < ", node.get_function().get_cut())
            else:
                print (node.get_variable, " > ", node.get_function().get_cut())

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
            print ("Output Reached after 1st node")
            return
        #Get the nodes that stem from the input node
        tmp = self.seed
        # print (tmp.get_weight())
        if tmp == None:
            print ("network->create_network: Could not create seed. Exiting...")
            return
        layer = self.seed.get_non_output_daughters()
        self.increment_layer()
        #While there are still new nodes to be created and we haven't reached...
        #... the max depth, keep building the network
        print ("New Nodes Created")

        while self.check_layer():
            tmp_new_nodes = []
            #Loop through all of the newly created node objects
            for node in layer:
                if node.is_output():
                    continue
                #Create the nodes that stem from this one
                if node.has_parent:
                    if node == node.get_parent().get_success_node():
                        win_df = node.get_parent().accept_pass_df()
                        lose_df = node.get_parent().accept_fail_df()
                    else:
                        win_df = node.get_parent().reject_pass_df()
                        lose_df = node.get_parent().reject_fail_df()
                else:
                    print_node(node)
                    print ("node has no parent")
                    continue
                node = self.prepare_node(win_df, lose_df, node)
                node.cut_dataset(win_df, lose_df)
                node = self.prepare_node(node.get_parent())
                new_inter_nodes += self.branch_node(node)
                if not node.get_success_node().is_output():
                    tmp_new_nodes = tmp_new_nodes.append(node.get_success_node())
                if not node.get_failure_node().is_output():
                    tmp_new_nodes = tmp_new_nodes.append(node.get_failure_node())
                #If this node isn't an output node, add the two new nodes to a temporary list
            if tmp_new_nodes == []:
                print("No new nodes created. exiting..")
                break
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
            #If the node is output there are no more nodes to add
            if node.is_output():
                return None
            #Set Up each node activation function
            yes_node = self.prepare_node(node.accept_pass_df,node.accept_fail_df, node.get_success_node())
            if yes_node == None:
                node.set_type(-1)
            no_node = self.prepare_node(node.reject_pass_df,node.reject_fail_df, node.get_failure_node())
            if no_node == None:
                node.set_type(-1)

            #Cut the datasets on each node
            if (node.is_output()):
                yes_node.cut_dataset(node.accept_pass_df, node.accept_fail_df)
                no_node.cut_dataset(node.reject_pass_df, node.reject_fail_df)
            self.node_count += 2
            #return the newly created nodes
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
        #find the best classification function for two arrays of the values of ONE variable...
        #... which correspond to winners and losers
        self.node_count += 1
        self.seed = self.neuron(self.node_count)
        self.seed.set_type(0)
        self.seed = self.prepare_node(win_df, lose_df, self.seed)
        self.increment_layer()
        self.seed.cut_dataset(win_df, lose_df)
        new_inter_nodes = self.branch_node(self.seed)
        #Function for neuron n now locked in with cut value and weight(=best cut ratio)
        return new_inter_nodes

    def create_output_node(self, node, direction, return_value):
        if direction:
            self.node_count += 1
            new_node = neuron(self.node_count)
            new_node.set_type(-1,return_value)
            node.set_success_path(new_node)
            # node.set_failure_path(None)
        else:
            self.node_count += 1
            new_node = neuron(self.node_count)
            new_node.set_type(-1,return_value)
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
            print ("*** NEW NODE --- LEVEL: ", self.get_current_layer(), " , ID: ", node.id, "***")
            if not (node.is_input()):
                print ("... PARENT: ", node.get_parent().id)
            else:
                print ("This is the Input Node")
            # print("Progress: |",flush = True)
            best_ratio = 0.0
            variables = list(win_df)
            usable_variable = False
            for variable in variables: # Need to create the variables array
                if (variable == "Species") | (self.check_variable_use(variable)):
                    continue# print "VARIABLE: ", variable
                usable_variable = True

                winners = win_df[variable]
                losers = lose_df[variable]
                # print "CheckPoint"
                result = node.find_best_activation_function(winners, losers, 50)
                if result == None:
                    continue
                #result[0] = best method index
                #result[1] = best cut value
                #result[2] = best cut ratio
                #result[3] = (Values greater than cut are winners) True OR False
                if result[2] > best_ratio:
                    best_results = result
                    var = variable
                #do for every variable to find which variable and method make the first node
            # print best_results
            if not usable_variable:
                return None
            node.set_function(best_results[0],best_results[1],best_results[2],best_results[3])
            print ("VARIABLE: ", var)
            self.add_to_used_variables(var)
            # print (result)
            # print (node.__str__())
            node.set_variable(var)
            node.set_type(self.get_current_layer())
            return node

    def total_weight_of_path(self, current_node):
        weight = 1.0
        # print (current_node.get_weight())
        while(1):
            # print ("blah",current_node.get_weight())
            weight = weight* current_node.get_weight()
            # print (weight)
            # print(current_node.get_type())
            if current_node.is_input():
                print("TOTAL WEIGHT OF PATH: ",weight)
                return weight
            else:
                current_node = current_node.get_parent()
                if current_node == None:
                    return weight



    def create_next_nodes(self, current_node):
        #Must build in checks to see if the next node should be an output
        # print (current_node.get_weight())
        #output when probability of success (of each node in this path multipled) reaches some satisfaction value
        if current_node == None:
            print ("network->create_next_nodes: Node Passed in is None")
            return False
        else:

            if (self.total_weight_of_path(current_node)) < 0.9:
                if current_node.has_result_dfs():
                    if current_node.accept_fail_df.empty | current_node.reject_pass_df.empty | current_node.accept_pass_df.empty | current_node.reject_fail_df.empty:
                        current_node.set_type(-1)
                        return False
                    self.node_count += 1
                    current_node.set_success_path(self.neuron(self.node_count))
                    current_node.get_success_node().set_parent(current_node,self.get_current_layer())
                    self.node_count += 1
                    current_node.set_failure_path(self.neuron(self.node_count))
                    current_node.get_failure_node().set_parent(current_node,self.get_current_layer())
                else:
                    print("network->create_next_nodes: Printing node information")
                    self.print_node(current_node)

            else:
                #Set the current node as output
                current_node.set_type(-1)
                return False
        return True

#End: Network Creation methods

#https://en.wikipedia.org/wiki/Activation_function
#classes of activation functions
    class neuron:
    #Neuron: Initialisation methods
        def universal_params(self):
            self.passind = 0
            self.failind = 1
        def __str__(self):
            if self.func_locked:
                return self.id, self.func.__str__()

        def __init__(self,ID):
            self.func_locked = False
            self.universal_params()
            self.id = ID
            self.split_data_set = False
            self.failure_node_exists = False
            self.success_node_exists = False
            self.type_set = False
            self.has_parent = False

        
    #End: Neuron: Initialisation methods
        def create_daughter(self, direction, ID):
            node = neuron(ID)
            if direction:
                self.set_success_path(node)
            else:
                self.set_failure_path(node)
            return

        def get_non_output_daughters(self):
            if (self.failure_node_exists) & (self.success_node_exists):
                if not ((self.get_success_node().is_output()) & (self.get_failure_node().is_output())):
                    nodes = [self.get_success_node(), self.get_failure_node]
                elif (self.get_success_node().is_output()):
                    nodes = [self.get_failure_node()]
                elif (self.get_failure_node().is_output()):
                    nodes = [self.get_success_node()]
                else:
                     nodes = None
                return nodes
            else:
                print("Daughter nodes not set. ID: ", self.id)
                return None

    #Network methods
        def set_parent(self, parent, layer):
            self.parent = parent
            self.set_type(layer)
            self.has_parent = True

        def set_variable(self, variable):
            self.variable = variable
            self.var_lock = True
            return

        def get_variable(self):
            return self.variable

        def get_parent(self):
            if self.has_parent:
                if self.get_type() == "Input":
                    return None
                return self.parent
            else:
                return None

        def set_type(self, layer, result = None):
            self.type_set = True
            if layer == 0:
                self.type = "Input"
            elif layer == -1:
                self.type = "Output"
                self.return_val = result
            else:
                self.type = "Intermediate"
            return

        def has_result_dfs(self):
            if self.func_locked:
                if not hasattr(self.func, 'accept_pass_df'):
                    print ("neuron->has_result_dfs: No DF for accepted pass events")
                    return False
                if not hasattr(self.func, 'reject_pass_df'):
                    print ("neuron->has_result_dfs: No DF for rejected pass events")
                    return False
                if not hasattr(self.func, 'accept_fail_df'):
                    print ("neuron->has_result_dfs: No DF for accepted fail events")
                    return False
                if not hasattr(self.func, 'reject_fail_df'):
                    print ("neuron->has_result_dfs: No DF for rejected fail events")
                    return False
                return True
            else:
                print("neuron->has_result_dfs: Function not locked")
                return False

        def get_type(self):
            return self.type

        def is_input(self):
            if self.type == "Input":
                return True
            else:
                return False

        def is_output(self):
            if self.get_type == "Output":
                return True
            else:
                return False

        def set_function(self, func_num):
            if self.func_locked:
                print ("neuron->set_function: Function already set and cannot be changed.\n")
                print (self.__str__())
                return
            else:
                self.func = self.function(func_num)
                self.func_locked = True
                # print ("Node:", self.id, " Locked")
                return

        def set_function(self, func_num, cut, weight, not_inverted = True):
            if self.func_locked:
                print ("neuron->set_function2: Function already set and cannot be changed.\n")
                print (self.__str__())
                return
            else:
                self.func = self.function(func_num)
                self.func.set_cut(cut, not_inverted)
                self.func_locked = True
                self.set_weight(weight)
                # print ("Node:", self.id, " Locked")
                return

        def get_function(self):
            print (self.func)
            return self.func

        def set_weight(self,weight):
            self.weight = weight
            return

        def get_weight(self):
            return self.weight

        def evaluate(self,x):
            if self.func_locked:
                return self.func.evaluate(x)
            else:
                print ("Function not locked, lock one in or try 'function::evaluate()'\nie. self.func must exist")
                return None

        #Check the best attribute function for pass/fail with value x
        def passes_cut(self, x):
            if self.func_locked:
                val  = self.evaluate(x)
                if val != None:
                    if self.func.cutdir == True:
                        if val > self.func.cut:
                            return True
                        else:
                            return False
                    else:
                        if val < self.func.cut:
                            return True
                        else:
                            return False
                else:
                    print ("Error: neuron->passes_cut: Could not evaluate function")
                    return None
            else:
                print ("Error: neuron->passes_cut: Function must be locked such that self.func exists")
                return None

        def set_success_path(self, next_neuron):
            self.success_node = next_neuron
            self.success_node_exists = True
            next_neuron.set_parent(self)
            return

        def set_failure_path(self, next_neuron):
            self.failure_node_exists = True
            self.failure_node = next_neuron
            next_neuron.set_parent(self)
            return

        def check_paths(self):
            if self.failure_node_exists & self.success_node_exists:
                return True
            else:
                return False

        def get_success_node(self):
            if self.success_node_exists:
                return self.success_node
            else:
                print ("neuron->get_success_node: Success node does not exist")
                return None

        def get_failure_node(self):
            if self.failure_node_exists:
                return self.failure_node
            else:
                print ("neuron->get_failure_node: Failure node does not exist")
                return None

        # Pass value to the next node,
        # if this is an output node, returns True of success and False for fail
        # otherwise, gets the result of pass_on() for the next node.
        # once an output node is reached, the result is returned
        def pass_on(self, entry):
            print("ID: ",self.id, self.get_type(), self.get_function().__str__())
            print(self.get_variable())
            value = entry[self.get_variable()]
            check = self.passes_cut(value)
            print("Passes Cut:", check)
            if check != None:
                if self.type != "Output":
                    if self.check_paths():
                        if check == True:
                            final = self.success_node.pass_on(entry)
                        elif check == False:
                            final = self.failure_node.pass_on(entry)
                        return final
                    else:
                        print ("Error: neuron->pass_on: Path not complete. Exiting...")
                        return None
                else:
                    return check
            else:
                print ("Error: neuron->pass_on: Couldn't Analyse Cut. Exiting...")
                return None


    #Neuron: End Network methods

    #Neuron: Training methods

        def find_best_activation_function(self, pass_array, fail_array, step_precision = 10):
            if(array_length_check(pass_array,fail_array)):
                results = []
                best_passer = -1
                best_failer = -1
                passer_ent = [0,0]
                failer_ent = [0,0]
                # print pass_array, fail_array
                for method in range(17):
                    highest_ratio = 0
                    lowest_ratio = 1.0
                    #Get the number of successfully catagorized events for each set
                    func = self.function(method)
                    results = func.find_best_cut(pass_array, fail_array, step_precision)
                    del func
                    if results == None:
                        print ("neuron->find_best_activation_function: Exiting...")
                        return None
                    if results[1] > highest_ratio:
                        highest_ratio = results[1]
                        cut_val_high = results[0]
                        best_method_high = method
                    elif results[1] < lowest_ratio:
                        lowest_ratio = results[1]
                        cut_val_low = results[0]
                        best_method_low = method
                low_inverse = (1 - lowest_ratio)
                if highest_ratio > low_inverse:
                    best_ratio = highest_ratio
                    cut_val = cut_val_high
                    best_method = best_method_high
                    dont_invert = True
                else:
                    best_ratio = low_inverse
                    cut_val = cut_val_low
                    best_method = best_method_low
                    dont_invert = False
                # self.set function(best_method, cut_val, dont_invert)

                return best_method, cut_val, best_ratio, dont_invert
            else:
                print ("neuron->find_best_activation_function: exiting...")
                return None

        def get_pass_list(self,df):
            pass_list = []
            fail_list = []
            for i in df.index:
                # print "line 367: ", df[self.variable][i]
                if self.passes_cut(df[self.variable][i]):
                    pass_list.append(i)
                else:
                    fail_list.append(i)
            return (pass_list, fail_list)

        def set_pass_dfs(self, list1, list2):


        def cut_dataset(self, win_df, lose_df):
            lists = self.get_pass_list(win_df)
            if (lists[0] == []):
                self.accept_pass_df = DataFrame({'none':[]})
                if not (self.accept_pass_df).empty():
                    print ("NOT EMPTY")
                print ("accept_pass_df set None")
                # self.set_type(-1,False)
                # return
            else:
                self.accept_pass_df = win_df.drop(lists[0]).reset_index()
                print ("accept_pass_df set")
            if lists[1] == []:
                self.reject_pass_df = pd.DataFrame({'none':[]})
                if (self.reject_pass_df).empty():
                    print ("NOT EMPTY")
                print ("reject_pass_df set None")
                # self.set_type(-1,True)
                # return
            else:
                self.reject_pass_df = win_df.drop(lists[1]).reset_index()
                print ("reject_pass_df set")

            lists2 = self.get_pass_list(lose_df)
            if (lists2[0] == []):
                self.accept_fail_df = None
                print ("accept_fail_df set None")
            else:
                self.accept_fail_df = lose_df.drop(lists2[0]).reset_index()
                print ("accept_fail_df set")

                # self.set_type(-1,True)
                # return
            if lists2[1] == []:
                self.reject_fail_df = None
                print ("reject_fail_df set None")
            else:
                self.reject_fail_df = lose_df.drop(lists2[1]).reset_index()
                print ("reject_fail_df set")

                # self.set_type(-1,False)
                # return
            # if self.accept_pass_df.empty:
            #     print("empty")
            # if self.reject_pass_df.empty:
            #     print("empty")
            # print self.accept_pass_df, self.reject_pass_df
            # if self.accept_fail_df.empty:
            #     print("empty")
            # if self.reject_fail_df.empty:
            #     print("empty")
            self.split_data_set = True
            return self

    #End: Neuron: Training methods

        class function:
            #Initialisation methods: class: function
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
                    print ("function number invalid, range = [0,17]")
                    self.func_lock = False
                return
        #End: Initialisation methods: class: function
            def __str__(self):
                return self.func.__str__()
        # Function Evaluation methods
            def set_cut(self, cut, greater_than = True):
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


        #End: Function Evaluation methods

        #Training methods
            #Returns two arrays of values after the initial values have been passed through the activation function
            def apply_function(self, pass_array, fail_array):
                if(array_length_check(pass_array,fail_array)):
                    if self.func_lock:
                        pass_results = []
                        fail_results = []
                        # print (pass_array)
                        for i in range(len(pass_array)):
                            # print("apply_function",i,pass_array[i])
                            pass_results.append(self.func.evaluate(pass_array[i]))
                            fail_results.append(self.func.evaluate(fail_array[i]))
                            # print("loop\n\n")
                        return(pass_results, fail_results)
                    else:
                        print ("Error: neuron->function->apply_function: Function must be set and locked")
                else:
                    print ("neuron->function->apply_function: exiting...")
                    return None

            def get_function_range(self, pass_output, fail_output):
                if(array_length_check(pass_output,fail_output)):
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
                    myrange = [mini,maxi]
                    return myrange
                else:
                    print ("neuron->function->get_function_range: exiting...")
                    return None

            def plot_hist(self, pass_array, fail_array):
                output = self.apply_function(pass_array,fail_array)
                plot_hist(output[0],output[1])
                return
            #Finds the best cut to use for a function which is locked in.
            #Iterates over the range of output for the function with number of iterations = step_number
            # Maximises the ratio of correctly identified points to incorrectly identified
            # Returns the best cut value and the ratio of pass/fail (including both sets)
            # Means that to find the best method only requires comparing output pf this function for each method
            def find_best_cut(self, pass_array, fail_array, step_number = 10):
                if(array_length_check(pass_array,fail_array)):
                    self.passind = 0
                    self.failind = 1
                    func_output = self.apply_function(pass_array, fail_array)
                    # print (func_output)
                    myrange = self.get_function_range(func_output[self.passind],func_output[self.failind])
                    # print (myrange[0],myrange[1])
                    step_size = float((myrange[1]-myrange[0])/step_number)
                    best_ratio = 0
                    for i in range(step_number):
                        cut_val = i*step_size
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
                        ratio = float(correct/(correct+incorrect))
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_cut = cut_val
                    return (cut_val, ratio)
                else:
                    print ("neuron->function->find_best_cut: exiting...")
                    return None
        #End Trining methods

        # Individual Function Classes
            #Function Member classes. Each function instance can only...
            #... invoke an instance of one of these classes
            #Identity function f(x) = x
            class identity:
                def __init__(self):
                    self.value = 1
                    self.range = [-float("inf"),float("inf")]
                def __str__(self):
                    return "Function: Identity"
                def setValue(self,value):
                    self.value = value
                def evaluate(self,val = 0):
                    return self.value*val
                def tweak_param(self,up):
                    if up:
                        self.value = self.value*1.01
                    else:
                        self.value = self.value*0.99


            class step:
                #Step Function f(x) = 0, x < 0
                #              f(x) = 1, x >= 0
                def __init__(self):
                    self.step = 0
                    self.range = [0,1]
                    return
                def __str__(self):
                    return "Function: Step Function"
                def setStep(self,value):
                    self.step = value
                    return
                def evaluate(self,val):
                    if val < self.step:
                        return 0
                    else:
                        return 1
                def tweak_param(self,up):
                    if up:
                        self.step += 1
                    else:
                        self.step -= 1
                    return

            class soft_step:
                #Soft step function f(x) = 1 / (1 + exp(-ax))
                def __init__(self):
                    self.expo = 1
                    self.range = [0,1]
                def __str__(self):
                    return "Function: Soft Step Function"
                def setExponent(self,value):
                    self.expo = value
                def evaluate(self,value):
                    return 1.0/(1+np.exp(-self.expo*value))

            class tanh:
                #Hyperbolic tanh function f(x) = tannh(ax)
                def __init__(self):
                    self.expo = 1
                    self.range = [-1,1]
                def __str__(self):
                    return "Function: tanh(x)"
                def setExponent(self,value = 2):
                    self.expo = value
                def evaluate(self,value):
                    return np.tan(self.expo*value)

            class arctan:
                #Inverse tan function f(x) = arctan(ax)
                def __init__(self):
                    self.coeff = 1
                    self.range = [-0.5*np.pi,0.5*np.pi]
                def __str__(self):
                    return "Function: arctan(x)"
                def setParam(self, value):
                    self.coeff = value
                def evaluate(self, value):
                    return np.arctan(self.coeff*value)

            class softsign:
                #Soft sign function f(x) = x / (1 + |x|)
                def __init__(self):
                    self.coeff = 1
                    self.range = [-1,1]
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
                #Rectified Linear Unit f(x) = 0, x < 0
                #                      f(x) = ax, x >= 0
                def __init__(self):
                    self.coeff = 1
                    self.range = [0,float("inf")]
                def __str__(self):
                    return "Function: ReLU"
                def setParam(self, value):
                    self.coeff = value
                def evaluate(self, value):
                    if value < 0:
                        return 0
                    else:
                        return self.coeff*value

            class leakyReLU:
                #Leaky ectified Linear Unit f(x) = 0.01ax, x < 0
                #                           f(x) = ax,     x >= 0
                def __init__(self):
                    self.coeff = 1
                    self.coeff2 = 1
                    self.range = [-float("inf"),float("inf")]
                def __str__(self):
                    return "Function: leakyReLU"
                def setParam(self, value, value2):
                    self.coeff = value
                    self.coeff2 = value2
                def evaluate(self, value):
                    if value < 0:
                        return 0.01 * self.coeff2 * value
                    else:
                        return self.coeff*value

            class PReLU:
                #parametric rectified Linear Unit f(x) = ax, x < 0
                #                           f(x) = bx, x >= 0
                def __init__(self):
                    self.coeff = 1
                    self.coeff2 = 1
                    self.range = [-float("inf"),float("inf")]
                def __str__(self):
                    return "Function: PReLU"
                def setParam(self, value, value2):
                    self.coeff = value
                    self.coeff2 = value2
                def evaluate(self, value):
                    if value < 0:
                        return self.coeff2 * value
                    else:
                        return self.coeff*value

            class ELU:
                #Exponential linear unit
                def __init__(self, value):
                    self.coeff = value
                    self.range = [-value,float("inf")]
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
                #S-shaped rectified linear activation unit
                def __init__(self):
                    self.tl = 0
                    self.tr = 1
                    self.al = 1
                    self.ar = 1
                    self.range = [-float("inf"),float("inf")]
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
                #Adaptive piecewise linear
                def __init__(self):
                    self.max = 1
                    self.range = [-float("inf"),float("inf")]
                def __str__(self):
                    return "Function: APL"
                def evaluate(self, x):
                    return 0
                #to be completed

            class soft_plus:
                # f(x) = ln ( 1 + exp(x) )
                def __init__(self):
                    self.range = [0,float("inf")]
                def __str__(self):
                    return "Function: Soft Plus"
                def evaluate(self, x):
                    return np.log( 1 + np.exp(x))

            class bent_identity:
                # f(x) = 0.5 * (sqrt(x^2 + 1) - 1) + x
                def __init__(self):
                    self.range = [-float("inf"),float("inf")]
                def __str__(self):
                    return "Function: Bent Identity"
                def evaluate(self, x):
                    return 0.5 * (np.sqrt(x*x + 1) - 1) + x

            class soft_exponential:
                def __init__(self, a):
                    self.a = a
                    self.range = [-float("inf"),float("inf")]
                    return
                def __str__(self):
                    return "Function: Soft Exponential"
                def evaluate(self, x):
                    if self.a < 0:
                        return (-1)*(np.log(1 - self.a * (x + self.a)))
                    elif self.a == 0:
                        return x
                    else:
                        return self.a + (np.exp(self.a * x) - 1) / self.a

            class sinusoid:
                def __init__(self):
                    self.range = [-1,1]
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
                        return np.sin(x)/x

            class gaussian:
                def __init__(self):
                    self.range = [0,1]
                def __str__(self):
                    return "Function: Gaussian"
                def evaluate(self, x):
                    return np.exp((-1)*x*x)
        # End Individual Function Classes


def normalise_data(df):
    for i,j in enumerate(df):
        if j == "Species":
            # print ("String")
            continue
        # print (df[j])
        maxi = max(df[j])
        df[j] = df[j]/maxi
        # print (df[j])
        # break
    return df

df = pd.read_csv("Iris.csv")
df = normalise_data(df)
net = network(5)
nwin_df = df[df["Species"]=="Iris-setosa"].reset_index()
nlose_df = df[df["Species"]!="Iris-setosa"].reset_index()
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
print("Cut Dataframe",len(nwin_df), len(nlose_df))

net.create_network(nwin_df, nlose_df)
print("network created")
net.visualise_tree()
#test_df =
event = df.iloc[130].drop("Species")
print (event)
answer = net.seed.pass_on(df.iloc[130].drop("Species"))
exit()
# print(answer)

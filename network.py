from node import neuron
import pandas as pd
import numpy as np
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
        self.seed = None
        self.failInd = 1
        self.passInd = 0
        return

    def add_to_used_variables(self, variable):
        if not self.used_variables:
            self.used_variables.append(variable)
        else:
            for i in self.used_variables:
                if i == variable:
                    return
            else:
                self.used_variables.append(variable)
        return

    def check_variable_use(self, variable):
        if not self.used_variables:
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
            if node.cut_inverted():
                print(node.get_variable, " < ", node.get_cut())
            else:
                print(node.get_variable, " > ", node.get_cut())
            print("Weight: ", node.weight)
            if isinstance(node.accept_pass_df, pd.DataFrame) & isinstance(node.accept_fail_df, pd.DataFrame):
                print("Accepted: Winners: ", len(node.accept_pass_df), "Losers: ", len(node.accept_fail_df))
            elif isinstance(node.accept_pass_df, pd.DataFrame):
                print("Accepted: Winners: ", len(node.accept_pass_df), "Losers: ", 0)
            elif isinstance(node.accept_fail_df, pd.DataFrame):
                print("Accepted: Winners: ", 0, "Losers: ", len(node.accept_fail_df))
            if isinstance(node.reject_pass_df, pd.DataFrame) & isinstance(node.reject_fail_df, pd.DataFrame):
                print("Rejected: Winners: ", len(node.reject_pass_df), "Losers: ", len(node.reject_fail_df))
            elif isinstance(node.reject_pass_df, pd.DataFrame):
                print("Rejected: Winners: ", len(node.reject_pass_df), "Losers: ", 0)
            elif isinstance(node.reject_fail_df, pd.DataFrame):
                print("Rejected: Winners: ", 0, "Losers: ", len(node.accept_fail_df))


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

    def create_network(self, win_df, lose_df, step_size):
        print("Creating First Node")
        self.create_first_node(win_df, lose_df, step_size)
        # print (self.seed.get_weight())
        # self.seed.get_function().plot_hist(win_df, lose_df)
        print("First Node Created")
        if self.seed.is_output():
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
            print(" --- NEW LAYER: ", self.current_layer, " ---")
            tmp_new_nodes = []
            new_inter_nodes = 0
            # Loop through all of the newly created node objects
            for node in layer:
                if not node:
                    raise Exception("Empty Node")
                if node.is_output():
                    continue
                # Create the nodes that stem from this one
                if node.has_parent:
                    if node is node.get_parent().get_success_node():
                        data = node.get_parent().get_data_sets(True)
                        # print (data)
                        if (not isinstance(data[0], pd.DataFrame)) and (not isinstance(data[1], pd.DataFrame)):
                            print("Good")
                            node.get_parent().set_type(-1, False)
                        elif not isinstance(data[0], pd.DataFrame):
                            node.set_type(-1, False)
                        elif not isinstance(data[1], pd.DataFrame):
                            node.set_type(-1, True)
                        # win_df = node.get_parent().accept_pass_df()   #ERROR THROWN HERE
                        # lose_df = node.get_parent().accept_fail_df()
                    else:
                        data = node.get_parent().get_data_sets(False)
                        if (not isinstance(data[0], pd.DataFrame)) and (not isinstance(data[1], pd.DataFrame)):
                            node.get_parent().set_type(-1,True)
                        elif not isinstance(data[0], pd.DataFrame):
                            node.set_type(-1, False)
                        elif not isinstance(data[1], pd.DataFrame):
                            node.set_type(-1, True)
                        # win_df = node.get_parent().reject_pass_df()
                        # lose_df = node.get_parent().reject_fail_df()
                else:
                    raise Exception("node has no parent")
                if (node.layer < 0) or (node.get_parent().layer < 0):
                    continue
                if not isinstance(data[0], pd.DataFrame) or not isinstance(data[1], pd.DataFrame):
                    raise ValueError("Emptry dataframes")
                # Find the best activation function and variable to use on the data that is fed into this node
                node = self.prepare_node(data[0], data[1], node, step_size)
                # Cut the dataset using the best function and applying it to the variable chosen in prepare_node()
                node.cut_dataset(data[0], data[1])
                # Create the nodes stemming from this one
                new_inter_nodes += self.branch_node(node)
                if node.get_success_node().layer > 0:
                    tmp_new_nodes = tmp_new_nodes.append(node.get_success_node())
                if not node.get_failure_node().layer > 0:
                    tmp_new_nodes = tmp_new_nodes.append(node.get_failure_node())
                    # If this node isn't an output node, add the two new nodes to a temporary list
            if not tmp_new_nodes:
                break
            else:
                layer = tmp_new_nodes
            self.increment_layer()
        return

    # def create_next_layer(self, node):
    #     if node == None:
    #         return None
    #     else:
    #         if node.is_output():
    #             return None
    #         # Create the nodes that are children of this node
    #         # print (node.get_weight())
    #
    #         new_layer = self.create_next_nodes(node)
    #
    #         if not new_layer:
    #             return None
    #         # If the node is output there are no more nodes to add
    #         if node.is_output():
    #             return None
    #         # Set Up each node activation ActFunctions
    #         yes_node = self.prepare_node(node.accept_pass_df, node.accept_fail_df, node.get_success_node())
    #         if yes_node == None:
    #             node.set_type(-1)
    #         no_node = self.prepare_node(node.reject_pass_df, node.reject_fail_df, node.get_failure_node())
    #         if no_node == None:
    #             node.set_type(-1)
    #
    #         # Cut the datasets on each node
    #         if (node.is_output()):
    #             yes_node.cut_dataset(node.accept_pass_df, node.accept_fail_df)
    #             no_node.cut_dataset(node.reject_pass_df, node.reject_fail_df)
    #         self.node_count += 2
    #         # return the newly created nodes
    #         return (yes_node, no_node)

    # returns the number of non-output nodes
    def branch_node(self, node):
        output_nodes = self.check_output(node)
        if output_nodes < 1:
            self.node_count += 1
            node.create_daughter(True, self.node_count, self.current_layer)
            self.node_count += 1
            node.create_daughter(False, self.node_count, self.current_layer)
            return 2
        elif output_nodes == 1:
            self.node_count += 1
            if node.success_node_exists:
                node.create_daughter(False, self.node_count, self.current_layer)
            else:
                node.create_daughter(True, self.node_count, self.current_layer)
            return 1
        else:
            return 0

    def create_first_node(self, win_df, lose_df, step_size = 10):
        # find the best classification ActFunctions for two arrays of the values of ONE variable...
        # ... which correspond to winners and losers
        self.node_count += 1
        self.seed = neuron(self.node_count)
        self.seed.set_type(0)
        self.seed = self.prepare_node(win_df, lose_df, self.seed, step_size)
        self.increment_layer()
        self.seed.cut_dataset(win_df, lose_df)
        new_inter_nodes = self.branch_node(self.seed)
        # Function for neuron n now locked in with cut value and weight(=best cut ratio)
        return new_inter_nodes

    def create_output_node(self, node, direction, return_value):
        if direction:
            print("New Output Node: Parent: ", node.id, ", Value: -WIN-")
            self.print_node(node)
            self.node_count += 1
            new_node = neuron(self.node_count)
            new_node.set_type(-1, return_value)
            node.set_success_path(new_node, self.current_layer)
            # node.set_failure_path(None)
        else:
            print("New Output Node: Parent: ", node.id, ", Value: -LOSE-")
            self.print_node(node)
            self.node_count += 1
            new_node = neuron(self.node_count)
            new_node.set_type(-1, return_value)
            node.set_failure_path(new_node, self.current_layer)
            # node.set_success_path(None)
        return

    def assess_ratio(self, win_df, lose_df):
        if len(win_df)/len(lose_df) < 0.05:
            return self.failInd
        elif len(win_df) / len(lose_df) > 0.95:
            return self.passInd
        else:
            return -1

    # Check if the node passed in should be an output node
    # use assess node to find which daughter node should bb e output
    def check_output(self, node):
        nodes_created = 0
        self.print_node(node)
        check = -1
        if node.layer < 0:
            return True
        # Check
        if isinstance(node.accept_pass_df, pd.DataFrame):
            if isinstance(node.accept_fail_df, pd.DataFrame):
                check = self.assess_ratio(node.accept_pass_df, node.accept_fail_df)
            if check == self.failInd or not isinstance(node.accept_fail_df, pd.DataFrame):
                self.create_output_node(node, True, False)
                nodes_created += 1
        if isinstance(node.accept_fail_df, pd.DataFrame):
            if check == self.passInd or not isinstance(node.accept_pass_df, pd.DataFrame):
                self.create_output_node(node, True, True)
                nodes_created += 1
        check = -1
        if isinstance(node.reject_pass_df, pd.DataFrame):
            if isinstance(node.reject_fail_df, pd.DataFrame):
                check = self.assess_ratio(node.reject_pass_df, node.reject_fail_df)
            if check == self.failInd or not isinstance(node.reject_fail_df, pd.DataFrame):
                self.create_output_node(node, False, False)
                nodes_created += 1
        if isinstance(node.reject_fail_df, pd.DataFrame):
            if check == self.passInd or not isinstance(node.reject_pass_df, pd.DataFrame):
                self.create_output_node(node, False, True)
                nodes_created += 1

        # elif not isinstance(node.accept_pass_df, pd.DataFrame):
        #     self.create_output_node(node, True, False)
        #     nodes_created += 1
        # elif not isinstance(node.accept_fail_df, pd.DataFrame):
        #     self.create_output_node(node, True, True)
        #     nodes_created += 1
        # elif not isinstance(node.reject_pass_df, pd.DataFrame):
        #     self.create_output_node(node, False, False)
        #     nodes_created += 1
        # elif not isinstance(node.reject_fail_df, pd.DataFrame):
        #     self.create_output_node(node, False, True)
        #     nodes_created += 1
        if nodes_created > 2:
            print("network->check_output: Node had no entries passed to it")
            self.print_node(node)
        return nodes_created

    # Calculate the best activation function for each variable and find which has best separation between training
    # samples
    def prepare_node(self, win_df, lose_df, node, step_size = 10):
        if node is None:
            raise ValueError("neuron object passed in has value None")
        else:
            if node.layer < 0:
                raise ValueError("neuron object passed in is an output node")
            print("*** NEW NODE --- LEVEL: ", self.get_current_layer(), " , ID: ", node.id, "***")
            if not (node.is_input()):
                print("... PARENT: ", node.get_parent().id)
            print("Progress: |", flush=True)
            best_ratio = 0.0
            variables = list(win_df)
            usable_variable = False
            # count = 0

            for variable in variables:
                # count += 1
                # printgit("Variable ", count , " of ", len(variables), flush=True)
                if ((win_df[variable].dtype != np.int64) & (win_df[variable].dtype != np.float64)) | (self.check_variable_use(variable)):
                    # print(win_df[variable].dtype, self.check_variable_use(variable))
                    continue
                #Anew variable has been found
                usable_variable = True
                #Get lists of this variable
                winners = win_df[variable]
                losers = lose_df[variable]
                # Find out the best activation function for this variable
                result = node.find_best_activation_function(winners, losers, step_size)
                # result[0/1/2/3] = best method index / best cut value / best cut value
                #                   / (Values greater than cut are winners) True OR False
                if result[2] > best_ratio:
                    best_ratio = result[2]
                    best_results = result
                    var = variable
            # do for every variable to find which variable and method
            if not usable_variable:
                return None
            # Lock the function to this node so that it's details cannot be changed
            node.set_function(best_results[0], best_results[1], best_results[2], best_results[3])
            print(node.__str__())
            print("VARIABLE: ", var)
            #Add variable to the list of ones used
            self.add_to_used_variables(var)
            #Set the variable and set type as either input of intermediate node
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
                if current_node is None:
                    return weight

    # def create_next_nodes(self, current_node):
    #     # Must build in checks to see if the next node should be an output
    #     # print (current_node.get_weight())
    #     # output when probability of success (of each node in this path multipled) reaches some satisfaction value
    #     if current_node == None:
    #         print("network->create_next_nodes: Node Passed in is None")
    #         return False
    #     else:
    #
    #         if (self.total_weight_of_path(current_node)) < 0.9:
    #             if current_node.has_result_dfs():
    #                 if current_node.accept_fail_df.empty | current_node.reject_pass_df.empty | current_node.accept_pass_df.empty | current_node.reject_fail_df.empty:
    #                     current_node.set_type(-1)
    #                     return False
    #                 self.node_count += 1
    #                 current_node.set_success_path(self.neuron(self.node_count))
    #                 current_node.get_success_node().set_parent(current_node, self.get_current_layer())
    #                 self.node_count += 1
    #                 current_node.set_failure_path(self.neuron(self.node_count))
    #                 current_node.get_failure_node().set_parent(current_node, self.get_current_layer())
    #             else:
    #                 print("network->create_next_nodes: Printing node information")
    #                 self.print_node(current_node)
    #
    #         else:
    #             # Set the current node as output
    #             current_node.set_type(-1)
    #             return False
    #     return True
    #
    # # End: Network Creation methods
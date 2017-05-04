from ActFunctions import *
import pandas as pd

# https://en.wikipedia.org/wiki/Activation_function
# classes of activation functions
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





# https://en.wikipedia.org/wiki/Activation_function
# classes of activation functions
class neuron(ActFunction):
    # Neuron: Initialisation methods
    def __str__(self):
        if self.func_locked:
            return self.id, self.func.__str__()

    def __init__(self, ID):
        self.failInd = 1
        self.passInd = 0
        self.func_locked = False
        self.layer = 1
        self.id = ID
        self.split_data_set = False
        self.failure_node_exists = False
        self.success_node_exists = False
        self.type_set = False
        self.has_parent = False
        self.parent = None
        self.return_val = None
        self.type = None
        self.variable = None
        self.var_lock = False
        self.weight = None
        self.success_node = None
        self.failure_node = None
        self.accept_fail_df = None
        self.accept_pass_df = None
        self.reject_fail_df = None
        self.reject_pass_df = None
        ActFunction.__init__(self)


        # End: Neuron: Initialisation methods

    def get_non_output_daughters(self):
        if self.failure_node_exists & self.success_node_exists:
            if not ((self.get_success_node().is_output()) & (self.get_failure_node().is_output())):
                nodes = [self.get_success_node(), self.get_failure_node()]
            elif self.get_success_node().is_output():
                nodes = [self.get_failure_node()]
            elif self.get_failure_node().is_output():
                nodes = [self.get_success_node()]
            else:
                nodes = None
            return nodes
        else:
            raise Exception("Daughter nodes not set. ID: ", self.id)

            # Network methods

    def create_daughter(self, direction, ID, layer):
        node = neuron(ID)
        if direction:
            self.set_success_path(node, layer)
        else:
            self.set_failure_path(node, layer)
        return

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

    def set_type(self, layer, result=None):
        self.type_set = True
        if layer == 0:
            self.type = "Input"
        elif layer == -1:
            self.layer = -1
            self.type = "Output"
            self.return_val = result
            self.success_node = None
            self.failure_node = None
        else:
            self.type = "Intermediate"
        return

    def has_result_dfs(self):
        if self.func_locked:
            if not hasattr(self.func, 'accept_pass_df'):
                print("neuron->has_result_dfs: No DF for accepted pass events")
                return False
            if not hasattr(self.func, 'reject_pass_df'):
                print("neuron->has_result_dfs: No DF for rejected pass events")
                return False
            if not hasattr(self.func, 'accept_fail_df'):
                print("neuron->has_result_dfs: No DF for accepted fail events")
                return False
            if not hasattr(self.func, 'reject_fail_df'):
                print("neuron->has_result_dfs: No DF for rejected fail events")
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
        if self.layer < 0:
            return True
        else:
            return False

    def set_function(self, func_num, cut, weight, not_inverted=True):
        if self.func_locked:
            print("neuron->set_function2: Function already set and cannot be changed.\n")
            print(self.__str__())
            return
        else:
            self.change_function(func_num)
            self.set_cut(cut, not_inverted)
            self.func_locked = True
            self.set_weight(weight)
            # print ("Node:", self.id, " Locked")
            return

    def get_function(self):
        print(self.func)
        return self.func

    def set_weight(self, weight):
        self.weight = weight
        return

    def get_weight(self):
        return self.weight

    def evaluate(self, x):
        if self.func_locked:
            return self.func.evaluate(x)
        else:
            raise ValueError("Function not locked, lock one in or try 'ActFunctions::evaluate()'\nie. self.func must "
                             "exist")

            # Check the best attribute ActFunctions for pass/fail with value x

    def passes_cut(self, x):
        if self.func_locked:
            val = self.evaluate(x)
            if val is not None:
                if self.cutdir:
                    if val > self.cut:
                        return True
                    else:
                        return False
                else:
                    if val < self.cut:
                        return True
                    else:
                        return False
            else:
                raise Exception("Error: neuron->passes_cut: Could not evaluate ActFunctions")

        else:
            raise Exception("Error: neuron->passes_cut: Function must be locked such that self.func exists")

    def set_success_path(self, next_neuron, layer):
        self.success_node = next_neuron
        self.success_node_exists = True
        next_neuron.set_parent(self, layer)
        return

    def set_failure_path(self, next_neuron, layer):
        self.failure_node_exists = True
        self.failure_node = next_neuron
        next_neuron.set_parent(self, layer)
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
            raise Exception("neuron->get_success_node: Success node does not exist")

    def get_failure_node(self):
        if self.failure_node_exists:
            return self.failure_node
        else:
            raise Exception("neuron->get_failure_node: Failure node does not exist")

    # Pass value to the next node,
    # if this is an output node, returns True of success and False for fail
    # otherwise, gets the result of pass_on() for the next node.
    # once an output node is reached, the result is returned
    def pass_on(self, entry):
        print("ID: ", self.id, self.get_type(), self.get_function().__str__())
        print(self.get_variable())
        value = entry[self.get_variable()]
        check = self.passes_cut(value)
        print("Passes Cut:", check)
        if check is not None:
            if self.type != "Output":
                if self.check_paths():
                    if check:
                        final = self.success_node.pass_on(entry)
                    elif not check:
                        final = self.failure_node.pass_on(entry)
                    return final
                else:
                    raise Exception("Error: neuron->pass_on: Path not complete. Exiting...")
            else:
                return check
        else:
            raise Exception("Error: neuron->pass_on: Couldn't Analyse Cut. Exiting...")


            # Neuron: End Network methods

            # Neuron: Training methods

    def find_best_activation_function(self, pass_array, fail_array, step_precision=10):
        # if array_length_check(pass_array, fail_array):
        highest_ratio = 0
        lowest_ratio = 1.0
        cut_val_high = None
        best_method_high = None
        cut_val_low = None
        best_method_low = None
        # print pass_array, fail_array
        for method in range(17):
            # Get the number of successfully catagorized events for each set
            self.change_function(method)
            results = self.find_best_cut(pass_array, fail_array, step_precision)
            if results[1] > highest_ratio:
                highest_ratio = results[1]
                cut_val_high = results[0]
                dont_invert = results[2]
                best_method_high = method
                # print("Method: ", method, ", ratio: ", highest_ratio)
        #     elif results[1] < lowest_ratio:
        #         lowest_ratio = results[1]
        #         cut_val_low = results[0]
        #         best_method_low = method
        # low_inverse = (1 - lowest_ratio)
        # if highest_ratio > low_inverse:
        best_ratio = highest_ratio
        cut_val = cut_val_high
        best_method = best_method_high
        # dont_invert = True
        # else:
        #     best_ratio = low_inverse
        #     cut_val = cut_val_low
        #     best_method = best_method_low
        #     dont_invert = False
        # self.set ActFunctions(best_method, cut_val, dont_invert)
        # self.set_function(best_method, cut_val, best_ratio, dont_invert)
        return best_method, cut_val, best_ratio, dont_invert
        # else:
        #     raise Exception("Array Length Error")

    def get_pass_list(self, df):
        pass_list = []
        fail_list = []
        for i in df.index:
            # print "line 367: ", df[self.variable][i]
            if self.passes_cut(df[self.variable][i]):
                pass_list.append(i)
            else:
                fail_list.append(i)
        return pass_list, fail_list

    # def set_pass_dfs(self, list1, list2):



    def get_data_sets(self, pass_data = True):
        if self.split_data_set:
            if pass_data:
                if isinstance(self.accept_pass_df, pd.DataFrame):
                    if isinstance(self.accept_fail_df, pd.DataFrame):
                        return self.accept_pass_df, self.accept_fail_df
                    return self.accept_pass_df, None
                if isinstance(self.accept_fail_df, pd.DataFrame):
                    return None, self.accept_fail_df
                return None, None
            else:
                if isinstance(self.reject_pass_df, pd.DataFrame):
                    if isinstance(self.reject_fail_df, pd.DataFrame):
                        return self.reject_pass_df, self.accept_fail_df
                    return self.reject_pass_df, None
                if isinstance(self.reject_fail_df, pd.DataFrame):
                    return None, self.reject_fail_df
                return None, None
        else:
            raise Exception("Data not been split")


    def cut_dataset(self, win_df, lose_df):
        if isinstance(win_df, pd.DataFrame):
            lists = self.get_pass_list(win_df)
        else:
            if win_df.empty():
                raise Exception("Empty win df. Add allowance for this")
        if not lists[0]:
            self.accept_pass_df = None
            # if not (self.accept_pass_df).empty():
            #     print("NOT EMPTY")
            print("accept_pass_df set None")
            # self.set_type(-1,False)
            # return
        else:
            self.accept_pass_df = win_df.drop(lists[0]).reset_index(drop=True)
            print("accept_pass_df set")
        if not lists[1]:
            self.reject_pass_df = None
            # if (self.reject_pass_df).empty():
            #     print("NOT EMPTY")
            # print("reject_pass_df set None")
            # self.set_type(-1,True)
            # return
        else:
            self.reject_pass_df = win_df.drop(lists[1]).reset_index(drop=True)
            print("reject_pass_df set")

        if isinstance(lose_df, pd.DataFrame):
            lists2 = self.get_pass_list(lose_df)
        else:
            if lose_df.empty():
                raise Exception("Empty win df. Add allowance for this")
        if not lists2[0]:
            self.accept_fail_df = None
            print("accept_fail_df set None")
        else:
            self.accept_fail_df = lose_df.drop(lists2[0]).reset_index(drop=True)
            print("accept_fail_df set")

            # self.set_type(-1,True)
            # return
        if not lists2[1]:
            self.reject_fail_df = None
            print("reject_fail_df set None")
        else:
            self.reject_fail_df = lose_df.drop(lists2[1]).reset_index(drop=True)
            print("reject_fail_df set")

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

#
# class neuron(ActFunctions):
#     # Neuron: Initialisation methods
#     def universal_params(self):
#         self.passind = 0
#         self.failind = 1
#
#     def __str__(self):
#         if self.func_locked:
#             return self.id, self.func.__str__()
#
#     def __init__(self, ID):
#         self.func_locked = False
#         self.universal_params()
#         self.id = ID
#         self.split_data_set = False
#         self.failure_node_exists = False
#         self.success_node_exists = False
#         self.type_set = False
#         self.has_parent = False
#
#
#         # End: Neuron: Initialisation methods
#
#     def create_daughter(self, direction, ID):
#         node = neuron(ID)
#         if direction:
#             self.set_success_path(node)
#         else:
#             self.set_failure_path(node)
#         return
#
#     def get_non_output_daughters(self):
#         if (self.failure_node_exists) & (self.success_node_exists):
#             if not ((self.get_success_node().is_output()) & (self.get_failure_node().is_output())):
#                 nodes = [self.get_success_node(), self.get_failure_node]
#             elif (self.get_success_node().is_output()):
#                 nodes = [self.get_failure_node()]
#             elif (self.get_failure_node().is_output()):
#                 nodes = [self.get_success_node()]
#             else:
#                 nodes = None
#             return nodes
#         else:
#             print("Daughter nodes not set. ID: ", self.id)
#             return None
#
#             # Network methods
#
#     def set_parent(self, parent, layer):
#         self.parent = parent
#         self.set_type(layer)
#         self.has_parent = True
#
#     def set_variable(self, variable):
#         self.variable = variable
#         self.var_lock = True
#         return
#
#     def get_variable(self):
#         return self.variable
#
#     def get_parent(self):
#         if self.has_parent:
#             if self.get_type() == "Input":
#                 return None
#             return self.parent
#         else:
#             return None
#
#     def set_type(self, layer, result=None):
#         self.type_set = True
#         if layer == 0:
#             self.type = "Input"
#         elif layer == -1:
#             self.type = "Output"
#             self.return_val = result
#         else:
#             self.type = "Intermediate"
#         return
#
#     def has_result_dfs(self):
#         if self.func_locked:
#             if not hasattr(self.func, 'accept_pass_df'):
#                 print("neuron->has_result_dfs: No DF for accepted pass events")
#                 return False
#             if not hasattr(self.func, 'reject_pass_df'):
#                 print("neuron->has_result_dfs: No DF for rejected pass events")
#                 return False
#             if not hasattr(self.func, 'accept_fail_df'):
#                 print("neuron->has_result_dfs: No DF for accepted fail events")
#                 return False
#             if not hasattr(self.func, 'reject_fail_df'):
#                 print("neuron->has_result_dfs: No DF for rejected fail events")
#                 return False
#             return True
#         else:
#             print("neuron->has_result_dfs: Function not locked")
#             return False
#
#     def get_type(self):
#         return self.type
#
#     def is_input(self):
#         if self.type == "Input":
#             return True
#         else:
#             return False
#
#     def is_output(self):
#         if self.get_type == "Output":
#             return True
#         else:
#             return False
#
#     def set_function(self, func_num):
#         if self.func_locked:
#             print("neuron->set_function: Function already set and cannot be changed.\n")
#             print(self.__str__())
#             return
#         else:
#             self.func = self.ActFunctions(func_num)
#             self.func_locked = True
#             # print ("Node:", self.id, " Locked")
#             return
#
#     def set_function(self, func_num, cut, weight, not_inverted=True):
#         if self.func_locked:
#             print("neuron->set_function2: Function already set and cannot be changed.\n")
#             print(self.__str__())
#             return
#         else:
#             self.func = self.ActFunctions(func_num)
#             self.func.set_cut(cut, not_inverted)
#             self.func_locked = True
#             self.set_weight(weight)
#             # print ("Node:", self.id, " Locked")
#             return
#
#     def get_function(self):
#         print(self.func)
#         return self.func
#
#     def set_weight(self, weight):
#         self.weight = weight
#         return
#
#     def get_weight(self):
#         return self.weight
#
#     def evaluate(self, x):
#         if self.func_locked:
#             return self.func.evaluate(x)
#         else:
#             print("Function not locked, lock one in or try 'ActFunctions::evaluate()'\nie. self.func must exist")
#             return None
#
#             # Check the best attribute ActFunctions for pass/fail with value x
#
#     def passes_cut(self, x):
#         if self.func_locked:
#             val = self.evaluate(x)
#             if val != None:
#                 if self.func.cutdir == True:
#                     if val > self.func.cut:
#                         return True
#                     else:
#                         return False
#                 else:
#                     if val < self.func.cut:
#                         return True
#                     else:
#                         return False
#             else:
#                 print("Error: neuron->passes_cut: Could not evaluate ActFunctions")
#                 return None
#         else:
#             print("Error: neuron->passes_cut: Function must be locked such that self.func exists")
#             return None
#
#     def set_success_path(self, next_neuron):
#         self.success_node = next_neuron
#         self.success_node_exists = True
#         next_neuron.set_parent(self)
#         return
#
#     def set_failure_path(self, next_neuron):
#         self.failure_node_exists = True
#         self.failure_node = next_neuron
#         next_neuron.set_parent(self)
#         return
#
#     def check_paths(self):
#         if self.failure_node_exists & self.success_node_exists:
#             return True
#         else:
#             return False
#
#     def get_success_node(self):
#         if self.success_node_exists:
#             return self.success_node
#         else:
#             print("neuron->get_success_node: Success node does not exist")
#             return None
#
#     def get_failure_node(self):
#         if self.failure_node_exists:
#             return self.failure_node
#         else:
#             print("neuron->get_failure_node: Failure node does not exist")
#             return None
#
#     # Pass value to the next node,
#     # if this is an output node, returns True of success and False for fail
#     # otherwise, gets the result of pass_on() for the next node.
#     # once an output node is reached, the result is returned
#     def pass_on(self, entry):
#         print("ID: ", self.id, self.get_type(), self.get_function().__str__())
#         print(self.get_variable())
#         value = entry[self.get_variable()]
#         check = self.passes_cut(value)
#         print("Passes Cut:", check)
#         if check != None:
#             if self.type != "Output":
#                 if self.check_paths():
#                     if check == True:
#                         final = self.success_node.pass_on(entry)
#                     elif check == False:
#                         final = self.failure_node.pass_on(entry)
#                     return final
#                 else:
#                     print("Error: neuron->pass_on: Path not complete. Exiting...")
#                     return None
#             else:
#                 return check
#         else:
#             print("Error: neuron->pass_on: Couldn't Analyse Cut. Exiting...")
#             return None
#
#
#             # Neuron: End Network methods
#
#             # Neuron: Training methods
#
#     def find_best_activation_function(self, pass_array, fail_array, step_precision=10):
#         if (array_length_check(pass_array, fail_array)):
#             results = []
#             best_passer = -1
#             best_failer = -1
#             passer_ent = [0, 0]
#             failer_ent = [0, 0]
#             # print pass_array, fail_array
#             for method in range(17):
#                 highest_ratio = 0
#                 lowest_ratio = 1.0
#                 # Get the number of successfully catagorized events for each set
#                 func = self.ActFunctions(method)
#                 results = func.find_best_cut(pass_array, fail_array, step_precision)
#                 del func
#                 if results == None:
#                     print("neuron->find_best_activation_function: Exiting...")
#                     return None
#                 if results[1] > highest_ratio:
#                     highest_ratio = results[1]
#                     cut_val_high = results[0]
#                     best_method_high = method
#                 elif results[1] < lowest_ratio:
#                     lowest_ratio = results[1]
#                     cut_val_low = results[0]
#                     best_method_low = method
#             low_inverse = (1 - lowest_ratio)
#             if highest_ratio > low_inverse:
#                 best_ratio = highest_ratio
#                 cut_val = cut_val_high
#                 best_method = best_method_high
#                 dont_invert = True
#             else:
#                 best_ratio = low_inverse
#                 cut_val = cut_val_low
#                 best_method = best_method_low
#                 dont_invert = False
#             # self.set ActFunctions(best_method, cut_val, dont_invert)
#
#             return best_method, cut_val, best_ratio, dont_invert
#         else:
#             print("neuron->find_best_activation_function: exiting...")
#             return None
#
#     def get_pass_list(self, df):
#         pass_list = []
#         fail_list = []
#         for i in df.index:
#             # print "line 367: ", df[self.variable][i]
#             if self.passes_cut(df[self.variable][i]):
#                 pass_list.append(i)
#             else:
#                 fail_list.append(i)
#         return (pass_list, fail_list)
#
#     # def set_pass_dfs(self, list1, list2):
#
#     def cut_dataset(self, win_df, lose_df):
#         lists = self.get_pass_list(win_df)
#         if (lists[0] == []):
#             self.accept_pass_df = DataFrame({'none': []})
#             if not (self.accept_pass_df).empty():
#                 print("NOT EMPTY")
#             print("accept_pass_df set None")
#             # self.set_type(-1,False)
#             # return
#         else:
#             self.accept_pass_df = win_df.drop(lists[0]).reset_index()
#             print("accept_pass_df set")
#         if lists[1] == []:
#             self.reject_pass_df = pd.DataFrame({'none': []})
#             if (self.reject_pass_df).empty():
#                 print("NOT EMPTY")
#             print("reject_pass_df set None")
#             # self.set_type(-1,True)
#             # return
#         else:
#             self.reject_pass_df = win_df.drop(lists[1]).reset_index()
#             print("reject_pass_df set")
#
#         lists2 = self.get_pass_list(lose_df)
#         if (lists2[0] == []):
#             self.accept_fail_df = None
#             print("accept_fail_df set None")
#         else:
#             self.accept_fail_df = lose_df.drop(lists2[0]).reset_index()
#             print("accept_fail_df set")
#
#             # self.set_type(-1,True)
#             # return
#         if lists2[1] == []:
#             self.reject_fail_df = None
#             print("reject_fail_df set None")
#         else:
#             self.reject_fail_df = lose_df.drop(lists2[1]).reset_index()
#             print("reject_fail_df set")
#
#             # self.set_type(-1,False)
#             # return
#         # if self.accept_pass_df.empty:
#         #     print("empty")
#         # if self.reject_pass_df.empty:
#         #     print("empty")
#         # print self.accept_pass_df, self.reject_pass_df
#         # if self.accept_fail_df.empty:
#         #     print("empty")
#         # if self.reject_fail_df.empty:
#         #     print("empty")
#         self.split_data_set = True
#         return self
#
#         # End: Neuron: Training methods
#
#
# def normalise_data(df):
#     for i, j in enumerate(df):
#         if j == "Species":
#             # print ("String")
#             continue
#         # print (df[j])
#         maxi = max(df[j])
#         df[j] = df[j] / maxi
#         # print (df[j])
#         # break
#     return df
#
#
# df = pd.read_csv("Iris.csv")
# df = normalise_data(df)
# net = network(5)
# nwin_df = df[df["Species"] == "Iris-setosa"].reset_index()
# nlose_df = df[df["Species"] != "Iris-setosa"].reset_index()
# nwin_df = nwin_df.drop("Species", axis=1)
# nlose_df = nlose_df.drop("Species", axis=1)
# print(nwin_df)
# # exit()
# if len(nwin_df) != len(nlose_df):
#     # print("Cutting Dataframe",len(win_df), len(lose_df))
#     if len(nwin_df) > len(nlose_df):
#         nwin_df = nwin_df[:len(nlose_df)]
#     else:
#         nlose_df = nlose_df[:len(nwin_df)]
# # print(win_df)
# # print(lose_df)
# print("Cut Dataframe", len(nwin_df), len(nlose_df))
#
# net.create_network(nwin_df, nlose_df)
# print("network created")
# net.visualise_tree()
# # test_df =
# event = df.iloc[130].drop("Species")
# print(event)
# answer = net.seed.pass_on(df.iloc[130].drop("Species"))
# exit()
# # print(answer)

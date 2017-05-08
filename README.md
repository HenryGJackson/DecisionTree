# DecisionTree
Self-written algorithm for machine learning based on a decision tree.

## Activation Functions
Each node in the tree has an associated activation function. While my definition of activation function may not be correct,
what this means is that cuts are made on the values yielded by variables after they pass through the function.  

The activation function for each node is determined by looping through all of the functions defined in the "Activation Functions" package.
This is repeated for multiple variables and the variable and function that maximise the number of correctly classified
entries are chosen for the given node.

## Definitions:
&nbsp;&nbsp;-&nbsp;&nbsp;Winning Events: Events that should be classified as positive  
&nbsp;&nbsp;-&nbsp;&nbsp;Losing Events: Events that should be classified as negative  
&nbsp;&nbsp;-&nbsp;&nbsp;Passed/Accepted Events: Events that are chosen as positive by a given node  
&nbsp;&nbsp;-&nbsp;&nbsp;Failed/Rejected Events: Events that are chosen as negative by a given node  
&nbsp;&nbsp;-&nbsp;&nbsp;Node/Neuron: Confusingly, I have currently used two naming schemes for the nodes.
Neuron is technically incorrect hence my switch to node. This must be corrected.  
&nbsp;&nbsp;-&nbsp;&nbsp;Input Node: There is only one of these in the programme, all date must pass through this node.  
&nbsp;&nbsp;-&nbsp;&nbsp;Intermediate Node: Daughter nodes that do not giva a definitive output. They make some decision and pass
subsets of the data passed in to their own daughter nodes.  
&nbsp;&nbsp;-&nbsp;&nbsp;Output Nodes: Once a satisfactory confidence level or the maximum depth of the tree has been reached, 
output nodes are created which give a boolean output with no decision making.  
&nbsp;&nbsp;-&nbsp;&nbsp;Activation Function: The data at each node is passed through a function and a cut made on the output
of said function.  
&nbsp;&nbsp;-&nbsp;&nbsp;Node Weight: The weight of a node is given by the probability that an entry in the training dataset was 
classified correctly.  
&nbsp;&nbsp;-&nbsp;&nbsp;Node Ratio: The ratio of winning events to losing events passed in to the node. A ratio < 0.05 or 
ratio > 0.95 results in creation of an output node.

## Work to be completed:
&nbsp;&nbsp;-&nbsp;&nbsp;Results do not currently get carried through from the parent to the daughter nodes 
for the "losing dataframe"  
&nbsp;&nbsp;-&nbsp;&nbsp;Check whether the algorithm to store newly created NON-OUTPUT nodes properly works.  
&nbsp;&nbsp;-&nbsp;&nbsp;Create only one output node and link all intermediates that would otherwise be output to this node.  
&nbsp;&nbsp;-&nbsp;&nbsp;Alter definitions and names used to fit convention.  

"""
-------------------------------------------------------------------------------------
Using Python to Manually Build Out a Simple Neural Network for Classification
-------------------------------------------------------------------------------------
Gianluca Capraro
Created: June 2019
-------------------------------------------------------------------------------------
This script demonstrates manual creation of a simple neural network to mimic the
TensorFlow API. The purpose is to closely develop an understanding of the building 
blocks that make up a neural network prior to working with the TensorFlow library. This
script will use artifical data to attempt to classify points into 1 of 2 classes.
-------------------------------------------------------------------------------------
"""
#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Operation():
	"""
 	An Operation is a node in a 'Graph'. TensorFlow also utilizes the concept of a 'Graph'.
    
    This class will be inherited by others that will compute the specific
    operation, such as addition or matrix multiplication.
	"""

	def __init__(self, input_nodes = []):
		"""
		Initialize the Operation to be Performed
		Set up input nodes and where they will be output.
		"""
		self.input_nodes = input_nodes
		self.output_nodes = []

		for node in input_nodes:
			node.output_nodes.append(self)

	def compute(self):
		""" 
        Placeholder function that will be overwritten by the specific operation
        that inherits from this class.
        """
		pass


class addition(Operation):
    """
	Subclass of Operation that actually performs a specific addition calculation.
    """
    def __init__(self, x, y):
     
        super().__init__([x, y])

    def compute(self, x_var, y_var):
         
        self.inputs = [x_var, y_var]
        return x_var + y_var


class multiplication(Operation):
    """
	Subclass of Operation that actually performs a specific multiplication calculation.
    """
    def __init__(self, a, b):
        
        super().__init__([a, b])
    
    def compute(self, a_var, b_var):
         
        self.inputs = [a_var, b_var]
        return a_var * b_var


class matmul(Operation):
    """
	Subclass of Operation that actually performs matrix multiplication.
    """
    def __init__(self, a, b):
        
        super().__init__([a, b])
    
    def compute(self, a_mat, b_mat):
         
        self.inputs = [a_mat, b_mat]
        return a_mat.dot(b_mat)


class Placeholder():
    """
    Placeholder - an empty node that needs a value to be provided to compute output.
    """
    def __init__(self):
        """
		Initialize the placeholder
        """
        self.output_nodes = []
        
        _default_graph.placeholders.append(self)


class Variable():
    """
    Variable - a changeable parameter of the Graph.
    """
    def __init__(self, initial_value = None):
        """
		Initialize the variable
        """
        self.value = initial_value
        self.output_nodes = []
         
        _default_graph.variables.append(self)


class Graph():
    """
	Graph - a global variable that connects variables and placeholders to operations.
    """
    def __init__(self):
        
        self.operations = []
        self.placeholders = []
        self.variables = []
        
    def set_as_default(self):
        """
        Makes this Graph as the Global Default Graph
        """
        global _default_graph
        _default_graph = self


def postorder_traverse(operation):
    """ 
    PostOrder Traversal of Nodes ensures the computations in given operation
    are performed in the correct order.
    """
    postorder_nodes = []
    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        postorder_nodes.append(node)

    recurse(operation)
    return postorder_nodes


class Session:
    """
	Session where operations will be performed.
    """
    def run(self, operation, feed_dict = {}):
        """ 
        Operation: The operation that will be computed
        Feed_dict: Dictionary that maps placeholders to inputs
        """
        # Put nodes in correct order
        postorder_nodes = postorder_traverse(operation)
        
        for node in postorder_nodes:

            if type(node) == Placeholder:
                node.output = feed_dict[node]

            elif type(node) == Variable:
                node.output = node.value 

            else: 
            	# Perform Operation
                node.inputs = [input_node.output for input_node in node.input_nodes]
                node.output = node.compute(*node.inputs)
            
            # Convert any lists to numpy arrays
            if type(node.output) == list:
                node.output = np.array(node.output)
        
        # Return the result of the operation output
        return operation.output


"""
-------------------------------------------------------------------------------------
Test Operations and Session Classes
-------------------------------------------------------------------------------------
We will use the following formulas to test operations - z = Ax + b, z = 10x + 1
-------------------------------------------------------------------------------------
"""
#Test the multiplication and addition operations
#create graph object
g = Graph()
#set graph as default global
g.set_as_default()
#get variable values for A and b
A = Variable(10)
b = Variable(1)
#create placeholder for expected function x
x = Placeholder()
#perform operations using created classes
y = multiplication(A,x)
z = addition(y,b)

#create session instance 
session = Session()
#get the result by running our session
result = session.run(operation = z, feed_dict = {x:10})
print('\nTesting Addition & Multiplication Operations, Session Classes...')
print('For z = Ax + b, where z = 10x + 1, Find z for x = 10:')
#get result (should be = 101 in this example)
print(result)
print('\n')

#Test the matrix multiplication operation using the same process
g = Graph()
g.set_as_default()
A = Variable([[10,20],[30,40]])
b = Variable([1,1])
x = Placeholder()
y = matmul(A,x)
z = addition(y,b)

session = Session()
result = session.run(operation = z, feed_dict = {x:10})
print('Testing Matrix Multiplication Operation...')
print('For z = Ax + b, where z = 10x + 1, Find z for x = 10:')
#get result (should be [[101,201],[301,401]] in this example)
print(result)
print('\n')


"""
-------------------------------------------------------------------------------------
Classification Excercise
-------------------------------------------------------------------------------------
"""

class Sigmoid(Operation):
	"""
	Define Sigmoid function to be used for classification
	"""
	def __init__(self,z):
		super().__init__([z])

	def compute(self, z_val):
		return 1/(1+np.exp(-z_val))

#import make_blobs from sklearn to create sample dataset
from sklearn.datasets import make_blobs

#initialize and define our sample dataset
data = make_blobs(n_samples = 50,n_features=2,centers=2,random_state=75)

#separate the labels from the features
features = data[0]
labels = data[1]

#visualize the data, notice it is very separable
#manually draw line to separate classes
print('Showing Manually Separated Scatter Plot of Artificial Dataset...')
x = np.linspace(0,11,10)
y = -x + 5
plt.scatter(features[:,0],features[:,1],c=labels,cmap='coolwarm')
plt.plot(x,y)
plt.show()
print('\n')

#create graph object and set as default
g = Graph()
g.set_as_default()

#create placeholder for x, the features (expected array)
x = Placeholder()

#create w, our matrix for mmultiplication
w = Variable([1,1])

#create b, the bias we will add
b = Variable(-5)

#define function z
z = addition(matmul(w,x),b)

#pass result into sigmoid function to determine classification
a = Sigmoid(z)

#create and run session to determine classification
#closer to 1, more certain about prediction
session = Session()
result1 = session.run(operation = a, feed_dict = {x:[8,10]})
result2 = session.run(operation = a, feed_dict = {x:[0,-10]})

print('Classification Results Below. Class either 1 or 0.')
print('\n')

#result for 1 should be 0.99
#this means our model is very sure that this point should be class 1
print("Classification of Result for x = [8,10]:")
print(result1)
print('\n')

#result for 2 should be very close to 0
#this means our model is very sure the point would belong to class 0
print("Classification of Result for x = [0,-10]:")
print(result2)
print('\n')


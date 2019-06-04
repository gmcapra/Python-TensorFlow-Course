"""
---------------------------------------------------------------------------------------
Use TensorFlow to Build a Simple Neural Network
---------------------------------------------------------------------------------------
Gianluca Capraro
Created: June 2019
---------------------------------------------------------------------------------------
The purpose of this script is to provide reference material for reviewing basic
TensorFlow syntax in Python, and also to demonstrate how TensorFlow can be used
to create a very basic Neural Network without the use of an estimator object.
---------------------------------------------------------------------------------------
"""
#import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

"""
---------------------------------------------------------------------------------------
TF Syntax Review
---------------------------------------------------------------------------------------
"""

#define the number of features and neurons
n_features = 10
n_dense_neurons = 3

#define our variables, weights, bias
x = tf.placeholder(tf.float32, (None, n_features))
W = tf.Variable(tf.random_normal([n_features,n_dense_neurons]))
b = tf.Variable(tf.ones([n_dense_neurons]))

#define your operations
xW = tf.matmul(x,W)
z = tf.add(xW,b)

#pass into your chosen activation function (using Sigmoid for this example)
a = tf.sigmoid(z)

#create the global variable initializer
init = tf.global_variables_initializer()

#run the operations in our tensorflow session
#initialize all variables
with tf.Session() as tf_sess:
	tf_sess.run(init)
	layer_out = tf_sess.run(a, feed_dict = {x: np.random.random([1,n_features])})


"""
---------------------------------------------------------------------------------------
Simple Regression Example
---------------------------------------------------------------------------------------
"""

#create some artificial data using numpy, add in noise
x_data = np.linspace(0,10,10) + np.random.uniform(-1.25, 1.25, 10)
y_label = np.linspace(0,10,10) + np.random.uniform(-1.25, 1.25, 10)

#plot the data to visualize relationship (should be linear)
print('\nShowing Scatter Plot of Artificial Data...')
sns.scatterplot(x_data,y_label)
plt.xlabel('x_data')
plt.ylabel('y_label')
plt.show()
print('\n')

#we want neural network to solve y = mx+b for our data
#initialize array with random value of m and random value of b
random_vals = np.random.rand(2)
print('Random X, Y (Slope, Intercept):')
m = tf.Variable(random_vals[0])
b = tf.Variable(random_vals[1])
print(random_vals)
print('\n')

#calculate the error
error = 0
for x,y in zip(x_data,y_label):

	#define predicted value
	y_hat = m*x + b

	#punish larger errors
	error += (y-y_hat)**2


#create optimizer using Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)

#use the optimizer to minimize error
train = optimizer.minimize(error)

#initialize global variables
init = tf.global_variables_initializer()

#start the session using 100 training steps
with tf.Session() as tf_sess:

	tf_sess.run(init)

	training_steps = 100

	for i in range(training_steps):
		tf_sess.run(train)

	final_slope, final_intercept = tf_sess.run([m,b])

#create test data
x_test = np.linspace(-1,11,10)

#get predicted line of best fit
y_pred_plot = final_slope*x_test + final_intercept

#plot the data set against the line of best fit, see how the model performed
print('Showing Artificial Dataset and Predicted Line of Best Fit...')
plt.plot(x_test, y_pred_plot, 'r')
plt.plot(x_data, y_label,'.')
plt.title('Predicted Line of Best Fit for Artificial Data')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
print('\n')

"""
Try playing with the # of training steps in our session. Does increasing or decreasing
this variable have an effect on the accuracy of our predicted line?
"""
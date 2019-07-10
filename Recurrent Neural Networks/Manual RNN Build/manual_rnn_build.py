"""
-------------------------------------------------------------------------------------
Using Python to Manually Build a Simple Example of a Recurrent Neural Network
-------------------------------------------------------------------------------------
Gianluca Capraro
Created: June 2019
-----------------------------------------------------------------------------------------
This script will demonstrate manual creation of a 3 neuron recurrent neural network layer
using the TensorFlow API. The purpose is to focus on the input format of the data
and to closely develop an understanding of the building blocks that make up a an RNN
prior to working in-depth with TensorFlow. 
-----------------------------------------------------------------------------------------
"""

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#define constants
num_inputs = 2
num_neurons = 3

#define placeholders
x0 = tf.placeholder(tf.float32, [None, num_inputs])
x1 = tf.placeholder(tf.float32, [None, num_inputs])

#define weight variables
Wx = tf.Variable(tf.random_normal(shape=[num_inputs,num_neurons]))
Wy = tf.Variable(tf.random_normal(shape=[num_neurons,num_neurons]))
b = tf.Variable(tf.zeros([1,num_neurons]))

#define graphs
y0 = tf.tanh(tf.matmul(x0,Wx) + b)
y1 = tf.tanh(tf.matmul(y0,Wy) + tf.matmul(x1,Wx) + b)

#initialize variables
init = tf.global_variables_initializer()

#create data
x0_batch = np.array([[0,1],[2,3],[4,5]])
x1_batch = np.array([[100,101],[102,103],[104,105]])

#run session
with tf.Session() as tf_sess:
    tf_sess.run(init)
    y0_output_vals, y1_output_vals = tf_sess.run([y0,y1], feed_dict = {x0:x0_batch, x1:x1_batch})


print('\ny0 Output Values:')
print(y0_output_vals)
print('\n')
print('\ny1 Output Values:')
print(y1_output_vals)
print('\n')









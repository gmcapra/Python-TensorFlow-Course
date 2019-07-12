"""
-------------------------------------------------------------------------------------
Autoencoders Practice Exercise
-------------------------------------------------------------------------------------
Gianluca Capraro
Created: July 2019
-----------------------------------------------------------------------------------------
To reduce the dimensionality of data, linear autoencoders can be used for principal
component analysis (PCA). This example will demonstrate the use of a linear autoencoder
to reduce the feature dimensions of artificially created data from 3D to 2D.
-----------------------------------------------------------------------------------------
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

#create sample data with makeblobs
data = make_blobs(n_samples = 100, n_features = 3, centers = 2, random_state = 101)

#scale the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[0])

#define columns of data
data_x = scaled_data[:,0]
data_y = scaled_data[:,1]
data_z = scaled_data[:,2]

#show 3D Mapping of x,y,z coordinate values
from mpl_toolkits.mplot3d import Axes3D
print("Showing 3D Representation of Artificial Data...")
figure = plt.figure()
axes = figure.add_subplot(111,projection='3d')
axes.scatter(data_x, data_y, data_z, c=data[1])
plt.show()
print('\n')

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
 
#define some model variables 
n_inputs = 3
n_hidden = 2
n_outputs = n_inputs
learning_rate = 0.01

#define placeholder
X = tf.placeholder(tf.float32,shape = [None, n_inputs])

#define layers
hidden = fully_connected(X,n_hidden,activation_fn=None)
outputs = fully_connected(hidden,n_outputs,activation_fn=None)

#define loss
loss = tf.reduce_mean(tf.square(outputs - X))

#define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

#initialize variables
init = tf.global_variables_initializer()


#define step count
n_steps = 1000

#create and run the session
with tf.Session() as tf_sess:
	tf_sess.run(init)

	for iter in range(n_steps):
		tf_sess.run(train,feed_dict={X:scaled_data})

	outputIn2d = hidden.eval(feed_dict={X:scaled_data})

#plot the resulting 2 dimensional output
print("Showing 2D Reduction of Data Performed by Linear Autoencoder...")
col_1 = outputIn2d[:,0]
col_2 = outputIn2d[:,1]
color_data = data[1]
plt.scatter(col_1,col_2,c=color_data)
plt.show()
print("\n")























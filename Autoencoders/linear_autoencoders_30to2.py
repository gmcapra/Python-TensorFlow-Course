"""
-------------------------------------------------------------------------------------
Linear Autoencoders for PCA - Reduce a 30 Dimensional Set to 2 Dimensions
-------------------------------------------------------------------------------------
Gianluca Capraro
Created: July 2019
-----------------------------------------------------------------------------------------
The purpose of this script is to create a Linear Autoencoder that can reduce a 30
dimension classification dataset into a 2 dimensional dataset.
-----------------------------------------------------------------------------------------
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#get the data and print some information
print("Anonymized Data Head and Info:")
df = pd.read_csv('anonymized_data.csv')
print(df.head())
print('\n')
print(df.info())
print('\n')

#scale the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df.drop('Label',axis=1))

#build out the linear autoencoder
from tensorflow.contrib.layers import fully_connected

#define model variables
n_inputs = 30
n_hidden = 2
n_outputs = n_inputs
learning_rate = 0.01

#create X placeholder
X = tf.placeholder(tf.float32, shape = [None, n_inputs])

#create layers
hidden = fully_connected(X,n_hidden,activation_fn=None)
outputs = fully_connected(hidden,n_outputs,activation_fn=None)

#create loss function
loss = tf.reduce_mean(tf.square(outputs - X))

#create optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

#create and run the session
#increase the steps for higher separation
print("\nReducing 30-Dimensional Dataset to 2-Dimensions...")
n_steps = 2500
with tf.Session() as tf_sess:
	tf_sess.run(init)

	for iter in range(n_steps):
		tf_sess.run(train,feed_dict={X:scaled_data})

#get the 2d output
with tf.Session() as tf_sess:
	tf_sess.run(init)
	outputIn2d = hidden.eval(feed_dict={X:scaled_data})

#show the 2d output
print("\nShowing Resulting 2D Output...")
plt.scatter(outputIn2d[:,0],outputIn2d[:,1],c=df['Label'])
plt.show()
print('\n')

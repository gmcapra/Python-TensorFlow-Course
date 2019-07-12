"""
-------------------------------------------------------------------------------------
Stacked Autoencoders with TensorFlow
-------------------------------------------------------------------------------------
Gianluca Capraro
Created: July 2019
-----------------------------------------------------------------------------------------
This script will use data from the MNIST Dataset, it is available from TensorFlow's public
examples and tutorials libraries. The purpose of this script is to demonstrate how to create 
stacked autoencoder and to help visualize how accurate the reconstructions coming out of 
the output layer of the model are.
-----------------------------------------------------------------------------------------
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#get the data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../03-Convolutional-Neural-Networks/MNIST_data/",one_hot=True)
tf.reset_default_graph() 

#define model parameters
n_inputs = 784 #28x28 pixels
neurons_hidden_1 = 392 #arbitrary choice
neurons_hidden_2 = 196 #arbitrary choice
neurons_hidden_3 = neurons_hidden_1 #where the decoder begins
n_outputs = n_inputs
learning_rate = 0.01

#define activation function
activation_func = tf.nn.relu

#create placeholder for X, unsupervised learning
X = tf.placeholder(tf.float32, shape = [None, n_inputs])

#need to initialize based on the size of the tensors
initializer = tf.variance_scaling_initializer()

#create weights
w1 = tf.Variable(initializer([n_inputs, neurons_hidden_1]), dtype=tf.float32)
w2 = tf.Variable(initializer([neurons_hidden_1, neurons_hidden_2]), dtype=tf.float32)
w3 = tf.Variable(initializer([neurons_hidden_2, neurons_hidden_3]), dtype=tf.float32)
w4 = tf.Variable(initializer([neurons_hidden_3, n_outputs]), dtype=tf.float32)

#create biases
b1 = tf.Variable(tf.zeros(neurons_hidden_1))
b2 = tf.Variable(tf.zeros(neurons_hidden_2))
b3 = tf.Variable(tf.zeros(neurons_hidden_3))
b4 = tf.Variable(tf.zeros(n_outputs))

#define hidden layers
hidden_layer_1 = activation_func(tf.matmul(X, w1) + b1)
hidden_layer_2 = activation_func(tf.matmul(hidden_layer_1, w2) + b2)
hidden_layer_3 = activation_func(tf.matmul(hidden_layer_2, w3) + b3)
output_layer = tf.matmul(hidden_layer_3, w4) + b4

#define loss function
loss = tf.reduce_mean(tf.square(output_layer - X))

#create optimizer and train
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

#initialize variable, create and run session
init = tf.global_variables_initializer()
saver = tf.train.Saver()
n_epochs = 5
batch_size = 150
with tf.Session() as tf_sess:
	tf_sess.run(init)

	for epoch in range(n_epochs):

		n_batches = mnist.train.num_examples // batch_size

		for iter in range(n_batches):
			X_batch, y_batch = mnist.train.next_batch(batch_size)
			tf_sess.run(train, feed_dict={X: X_batch})

		training_loss = loss.eval(feed_dict={X: X_batch})   
		print("Epoch {} Complete. Training Loss is {}".format(epoch,training_loss))

	saver.save(tf_sess, "./stacked_autoencoder.ckpt")    


"""
-----------------------------------------------------------------------------------------
Test the Autoencoder Output on the Test Data
-----------------------------------------------------------------------------------------
"""
#define test images autoencoder hasnt seen and run session
n_test_images = 10

with tf.Session() as tf_sess:
	saver.restore(tf_sess,"./stacked_autoencoder.ckpt")
	results = output_layer.eval(feed_dict={X:mnist.test.images[:n_test_images]})

#compare original images with their reconstructed versions
print("Showing Original MNIST Image vs. Reconstructed by Autoencoder...")
f, a = plt.subplots(2, 10, figsize=(20, 4))
for i in range(n_test_images):
	a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
	a[1][i].imshow(np.reshape(results[i], (28, 28)))

plt.show()
print('\n')





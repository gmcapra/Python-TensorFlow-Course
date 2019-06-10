"""
--------------------------------------------------------------------------------------
Build a Convolutional Neural Network to Classify Images from the MNIST Dataset
--------------------------------------------------------------------------------------
Gianluca Capraro
Created: June 2019
--------------------------------------------------------------------------------------
The purpose of this project is to demonstrate how TensorFlow can be used to build out 
a convolutional neural network. This CNN will be trained and used to make predictions 
from the MNIST handwritten numbers dataset.
--------------------------------------------------------------------------------------
"""

#import libraries
import tensorflow as tf
import pandas as pd

#import the MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data

#read in the dataset
data = input_data.read_data_sets("MNIST_data/",one_hot=True)

"""
--------------------------------------------------------------------------------------
Define Helper Functions
--------------------------------------------------------------------------------------
"""

def init_weights(shape):
	"""
	Initialize random weights for convolutional laters
	"""
	init_random_dist = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(init_random_dist)

def init_bias(shape):
	"""
	Initialize random biases for the layers
	"""
	init_bias_vals = tf.constant(0.1, shape=shape)
	return tf.Variable(init_bias_vals)

def conv2D(x, W):
	"""
	Create a 2D convolution using tensorflows built in function
	1 - Flatten the filter to a 2D matrix with shape [filter_height, filter_width, in_channels, output_channels]
	2 - Extract image patches from input tensor and form a virtual tensor of the shape
	3 - Right multiple the filter matrix and image patch vector for each patch
	"""
	return tf.nn.conv2d(
    					x,
    					W,
    					strides=[1, 1, 1, 1],
    					padding='SAME')

def max_pool_2x2(x):
	"""
	Use built in tensorflow function to perform max pooling on the input
	"""
	return tf.nn.max_pool(
						x,
						ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME')


def convolutional_layer(input_x, shape):
	"""
	Use the convert2d function to return convolutional layer that uses ReLu activation function.
	"""
	W = init_weights(shape)
	b = init_bias([shape[3]])
	return tf.nn.relu(conv2D(input_x, W) + b)

def normal_full_layer(input_layer, size):
	"""
	Return a normal fully connected layer.
	"""
	input_size = int(input_layer.get_shape()[1])
	W = init_weights([input_size, size])
	b = init_bias([size])
	return tf.matmul(input_layer, W) + b


"""
--------------------------------------------------------------------------------------
Build out the CNN
--------------------------------------------------------------------------------------
"""

#define placeholder for the data
x = tf.placeholder(tf.float32, shape = [None, 784])

#define the true labels placeholder
y_true = tf.placeholder(tf.float32, shape = [None, 10])

#define input layer
x_image = tf.reshape(x, [-1,28,28,1])

#define first convolutional layer 
cl_1 = convolutional_layer(x_image, shape = [6,6,1,32])
cl_1_pooling = max_pool_2x2(cl_1)

#pass first layer output into our second convolutional layer
cl_2 = convolutional_layer(cl_1_pooling, shape = [6,6,32,64])
cl_2_pooling = max_pool_2x2(cl_2)

#flatten out the result of second layer to create the full first layer
cl_2_flat = tf.reshape(cl_2_pooling, [-1,7*7*64])
layer_1_full = tf.nn.relu(normal_full_layer(cl_2_flat,1024))

#define the holdout probability and 
hold_probability = tf.placeholder(tf.float32)

#add dropout to prevent overfitting
full_1_dropout = tf.nn.dropout(layer_1_full, keep_prob = hold_probability)

#define the label predictions from the nn
y_predictions = normal_full_layer(full_1_dropout, 10)

#create a cross entropy loss function using built in tf functions
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_predictions))

#set up optimizer and train using the cross entropy loss function
optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001)
train = optimizer.minimize(cross_entropy)

"""
--------------------------------------------------------------------------------------
Define and Run the TensorFlow Session to Make Predictions using the CNN
--------------------------------------------------------------------------------------
"""

#declare global variable initializer
init = tf.global_variables_initializer()

#define number of steps to run
steps = 5000

#create and run the session (note: depending on your computer's speed, this can take a long time)
with tf.Session() as tf_sess:
	tf_sess.run(init)
	for i in range(steps):
		batch_x ,batch_y = data.train.next_batch(50)
		tf_sess.run(train,feed_dict={x:batch_x,y_true:batch_y,hold_probability:0.5})

		#print results and accuracy every 100 steps
		if i%100 == 0:
			print('Currently on step {}'.format(i))
			print('Accuracy is:')

			#test the model predictions against the true values
			matches = tf.equal(tf.argmax(y_predictions,1),tf.argmax(y_true,1))
			acc = tf.reduce_mean(tf.cast(matches,tf.float32))
			print(tf_sess.run(acc,feed_dict={x:data.test.images,y_true:data.test.labels,hold_probability:1.0}))
			print('\n')


"""
Within only a few hundred steps the model already surpasses 90% accuracy in making
predictions from the MNIST dataset. Running to completion, the CNN is trained to 
predict with > 98% accuracy.
"""


"""
--------------------------------------------------------------------------------------
Project: Create and Train a CNN to Make Classifications from the CIFAR-10 Dataset
--------------------------------------------------------------------------------------
Gianluca Capraro
Created: June 2019
--------------------------------------------------------------------------------------
The CIFAR-10 dataset, associated information, and setup code can be found here:
https://www.cs.toronto.edu/~kriz/cifar.html

The purpose of this project is to show another example of the use of TensorFlow to
build, train, and make predictions from a convolutional neural network. The CNN
built in this project will read in the images from the CIFAR-10 dataset and attempt to
classify them as either an airplane, automobile, bird, cat, deer, dog, frog, horse,
ship, or truck.
--------------------------------------------------------------------------------------
"""

#import libraries
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
--------------------------------------------------------------------------------------
Set Up the Data Using Code Provided by CIFAR 10 Data Providers
--------------------------------------------------------------------------------------
"""

#define the directory where the CIFAR data is stored
CIFAR_DIR = 'cifar-10-batches-py/'

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        cifar_dict = pickle.load(fo, encoding='bytes')
    return cifar_dict

dirs = ['batches.meta','data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch']    

all_data = [0,1,2,3,4,5,6]

for i,direc in zip(all_data,dirs):
    all_data[i] = unpickle(CIFAR_DIR+direc)

batch_meta = all_data[0]
data_batch1 = all_data[1]
data_batch2 = all_data[2]
data_batch3 = all_data[3]
data_batch4 = all_data[4]
data_batch5 = all_data[5]
test_batch = all_data[6]

X = data_batch1[b"data"] 
X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")

#show some test images to verify
print('\nShowing Test Image 1...')
plt.imshow(X[0])
plt.show()
print('\n')

print('Showing Test Image 2...')
plt.imshow(X[4])
plt.show()
print('\n')

print('Showing Test Image 3...')
plt.imshow(X[8])
plt.show()
print('\n')


"""
--------------------------------------------------------------------------------------
Define Helper Functions for Handling Data
--------------------------------------------------------------------------------------
"""
def one_hot_encode(vec, vals=10):
    """
    One-hot encode the 10- possible labels
    """
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out


class CifarHelper():

    """
	Code provided by class to help deal with getting the next batch.
    """

    def __init__(self):

        self.i = 0
        
        self.all_train_batches = [data_batch1,data_batch2,data_batch3,data_batch4,data_batch5]
        self.test_batch = [test_batch]
        
        self.training_images = None
        self.training_labels = None
        
        self.test_images = None
        self.test_labels = None
    
    def set_up_images(self):
        
        print("Setting Up Training Images and Labels")
        
        self.training_images = np.vstack([d[b"data"] for d in self.all_train_batches])
        train_len = len(self.training_images)
        
        self.training_images = self.training_images.reshape(train_len,3,32,32).transpose(0,2,3,1)/255
        self.training_labels = one_hot_encode(np.hstack([d[b"labels"] for d in self.all_train_batches]), 10)
        
        print("Setting Up Test Images and Labels")
        
        self.test_images = np.vstack([d[b"data"] for d in self.test_batch])
        test_len = len(self.test_images)
        
        self.test_images = self.test_images.reshape(test_len,3,32,32).transpose(0,2,3,1)/255
        self.test_labels = one_hot_encode(np.hstack([d[b"labels"] for d in self.test_batch]), 10)

        
    def next_batch(self, batch_size):
        x = self.training_images[self.i:self.i+batch_size].reshape(100,32,32,3)
        y = self.training_labels[self.i:self.i+batch_size]
        self.i = (self.i + batch_size) % len(self.training_images)
        return x, y


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
Building the CNN
--------------------------------------------------------------------------------------
"""

#define placeholder for the data
x = tf.placeholder(tf.float32, shape = [None, 32, 32, 3])

#define the true labels placeholder
y_true = tf.placeholder(tf.float32, shape = [None, 10])

#define first convolutional layer 
cl_1 = convolutional_layer(x, shape = [4,4,3,32])
cl_1_pooling = max_pool_2x2(cl_1)

#pass first layer output into our second convolutional layer
cl_2 = convolutional_layer(cl_1_pooling, shape = [4,4,32,64])
cl_2_pooling = max_pool_2x2(cl_2)

#flatten out the result of second layer to create the full first layer
cl_2_flat = tf.reshape(cl_2_pooling, [-1,8*8*64])
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
Run, Train, Evaluate the CNN
--------------------------------------------------------------------------------------
"""

#initialize instance of Cifar Helper class and set up the images
ch = CifarHelper()
ch.set_up_images()

#declare the global variable initializer
init = tf.global_variables_initializer()

#set the number of steps
n_steps = 5000

#create and run the session using cifar helper object to grab batches
with tf.Session() as tf_sess:
    tf_sess.run(tf.global_variables_initializer())

    for i in range(n_steps):
        batch = ch.next_batch(100)
        tf_sess.run(train, feed_dict={x: batch[0], y_true: batch[1], hold_probability: 0.5})
        
        #print results every 100 steps
        if i%100 == 0:
            
            print('Currently on step {}'.format(i))
            print('Accuracy is:')
            
            #evaluate predictions against true values
            matches = tf.equal(tf.argmax(y_predictions,1),tf.argmax(y_true,1))
            acc = tf.reduce_mean(tf.cast(matches,tf.float32))
            print(tf_sess.run(acc,feed_dict={x:ch.test_images,y_true:ch.test_labels,hold_probability:1.0}))
            print('\n')

"""
This model is capable of learning to classify the images with ~70% accuracy after training
for approximately 5,000 steps.
"""





"""
-------------------------------------------------------------------------------------
Using Python and Tensorflow to build a Recurrent Neural Network
-------------------------------------------------------------------------------------
Gianluca Capraro
Created: June 2019
-----------------------------------------------------------------------------------------
The purpose of this script is to create a simple RNN that attempts to predict a time
series that is shifted over 1 unit into the future. Additionally, this script will attempt
to generate new sequences given a seed series.
-----------------------------------------------------------------------------------------
"""

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


"""
-----------------------------------------------------------------------------------------
Create a class that can create data and generate batches
-----------------------------------------------------------------------------------------
"""

class TimeSeriesData():

	def __init__(self,num_points,xmin,xmax):

		self.xmin = xmin
		self.xmax = xmax
		self.num_points = num_points
		self.resolution = (xmax-xmin)/num_points
		self.x_data = np.linspace(xmin,xmax,num_points)
		self.y_true = np.sin(self.x_data)

	def ret_true(self,x_series):
		return np.sin(x_series)

	def next_batch(self,batch_size,steps,return_batch_ts=False):

		#grab a random starting point for batch
		rand_start = np.random.rand(batch_size,1)

		#convert to be on time series
		ts_start = rand_start * (self.xmax- self.xmin - (steps*self.resolution))

		#create batch time series on x axis
		batch_ts = ts_start + np.arange(0.0,steps+1) * self.resolution

		#create y data for time series x axis
		y_batch = np.sin(batch_ts)

		#format data for rnn
		if return_batch_ts:
			return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1) ,batch_ts
		else:
			return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1) 
        

#create some time series data
ts_data = TimeSeriesData(250,0,10)

#plot the data
print("\nShowing Time Series Data Plot...")
plt.plot(ts_data.x_data,ts_data.y_true)
plt.title("Time Series Data")
plt.show()
print('\n')

#set number of steps in batch (to be used for prediction steps)
num_time_steps = 30

#get the next batch data
y1,y2,ts = ts_data.next_batch(1,num_time_steps,True)

#show only batch section of plot data
print("\nShowing Time Series Batch Plot...")
plt.plot(ts.flatten()[1:],y2.flatten(),'*')
plt.title("Time Series Batch")
plt.show()
print('\n')

#show batch overlayed onto overall data
print("\nShowing Time Series Data and Batch Plot...")
plt.plot(ts_data.x_data,ts_data.y_true,label='Sin(t)')
plt.plot(ts.flatten()[1:],y2.flatten(),'*',label='Single Training Instance')
plt.legend()
plt.tight_layout()
plt.show()
print('\n')


"""
-----------------------------------------------------------------------------------------
A training instance and predicting a time series shifted by t+1
-----------------------------------------------------------------------------------------
"""

#define the training instance
train_inst = np.linspace(5,5 + ts_data.resolution * (num_time_steps + 1), num_time_steps + 1)

#plot the training instance
print("\nShowing Training Instance...")
plt.title("Training Instance", fontsize=15)
plt.plot(train_inst[:-1], ts_data.ret_true(train_inst[:-1]), "bo", markersize=15,alpha=0.5 ,label="instance")
plt.plot(train_inst[1:], ts_data.ret_true(train_inst[1:]), "ko", markersize=7, label="target")
plt.show()
print('\n')


"""
-----------------------------------------------------------------------------------------
Create the RNN Model
-----------------------------------------------------------------------------------------
"""

#define number of features: the time series
num_inputs = 1

#100 neuron layer, can be adjusted to observe different results
num_neurons = 100

#define output: the predicted time series
num_outputs = 1

#learning rate, 0.0001 default, can also be adjusted
learning_rate = 0.0001

#how many iterations (training steps), can also be adjusted
num_train_iterations = 2000

#Size of the batch
batch_size = 1

#define the placeholder values
X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])

#create RNN cell layer (try each one, compare performance)
cell = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.BasicRNNCell(num_units=num_neurons, activation=tf.nn.relu),
    output_size=num_outputs)

"""
--------------------------------------------------------------------------------------
cell = tf.contrib.rnn.OutputProjectionWrapper(
     tf.contrib.rnn.BasicLSTMCell(num_units=num_neurons, activation=tf.nn.relu),
     output_size=num_outputs)
--------------------------------------------------------------------------------------
n_neurons = 100
n_layers = 3
cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
           for layer in range(n_layers)])
--------------------------------------------------------------------------------------
cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_neurons, activation=tf.nn.relu)
--------------------------------------------------------------------------------------
n_neurons = 100
n_layers = 3
cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)
           for layer in range(n_layers)])
--------------------------------------------------------------------------------------
"""

#create dynamic RNN cell
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

#define loss function and optimizer
loss = tf.reduce_mean(tf.square(outputs - y)) # MSE
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

#initialize variables
init = tf.global_variables_initializer()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
#create saver
saver = tf.train.Saver()

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    
    for iteration in range(num_train_iterations):
        
        X_batch, y_batch = ts_data.next_batch(batch_size, num_time_steps)
        sess.run(train, feed_dict={X: X_batch, y: y_batch})
        
        if iteration % 100 == 0:
            
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(iteration, "\tMSE:", mse)
    
    # Save Model for Later
    saver.save(sess, "./rnn_time_series_model")

#create and run the session
with tf.Session() as tf_sess:                          
    saver.restore(tf_sess, "./rnn_time_series_model")   

    X_new = np.sin(np.array(train_inst[:-1].reshape(-1, num_time_steps, num_inputs)))
    y_pred = tf_sess.run(outputs, feed_dict={X: X_new})

#show the predictions and testing model data
print("Showing Testing Model...")
#plot the testing model
plt.title("Testing Model")
# Training Instance
plt.plot(train_inst[:-1], np.sin(train_inst[:-1]), "bo", markersize=15,alpha=0.5, label="Training Instance")
# Target to Predict
plt.plot(train_inst[1:], np.sin(train_inst[1:]), "ko", markersize=10, label="target")
# Models Prediction
plt.plot(train_inst[1:], y_pred[0,:,0], "r.", markersize=10, label="prediction")
plt.xlabel("Time")
plt.legend()
plt.tight_layout()
plt.show()
print('\n')

"""
Should be able to see clear improvement in prediction accuracy over the time steps.
"""


"""
-----------------------------------------------------------------------------------------
Generating New Sequences
-----------------------------------------------------------------------------------------
"""

print("Showing Generated Sequences...")

with tf.Session() as sess:
    saver.restore(sess, "./rnn_time_series_model")

    # Seed with zeros
    zero_seq_seed = [0. for i in range(num_time_steps)]
    for iteration in range(len(ts_data.x_data) - num_time_steps):
        X_batch = np.array(zero_seq_seed[-num_time_steps:]).reshape(1, num_time_steps, 1)
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        zero_seq_seed.append(y_pred[0, -1, 0])


plt.plot(ts_data.x_data, zero_seq_seed, "b-")
plt.plot(ts_data.x_data[:num_time_steps], zero_seq_seed[:num_time_steps], "r", linewidth=3)
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()
print('\n')

with tf.Session() as sess:
    saver.restore(sess, "./rnn_time_series_model")

    # Seed with Training Instance
    training_instance = list(ts_data.y_true[:30])
    for iteration in range(len(training_instance) -num_time_steps):
        X_batch = np.array(training_instance[-num_time_steps:]).reshape(1, num_time_steps, 1)
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        training_instance.append(y_pred[0, -1, 0])

plt.plot(ts_data.x_data, ts_data.y_true, "b-")
plt.plot(ts_data.x_data[:num_time_steps],training_instance[:num_time_steps], "r-", linewidth=3)
plt.xlabel("Time")
plt.show()
print('\n')







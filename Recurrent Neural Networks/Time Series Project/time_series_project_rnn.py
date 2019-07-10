"""
-------------------------------------------------------------------------------------
Neural Networks Course Project - Time Series RNN with Python and Tensorflow
-------------------------------------------------------------------------------------
Gianluca Capraro
Created: July 2019
-----------------------------------------------------------------------------------------
The purpose of this script is to create a Recurrent Neural Network that is able to 
generate predicted values from milk production data and compare these values to the
expected production.
-----------------------------------------------------------------------------------------
"""

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


"""
-----------------------------------------------------------------------------------------
Verify the Data
-----------------------------------------------------------------------------------------
"""

#print the data head
print('\nMonthly Milk Production Data Head:')
data = pd.read_csv('monthly-milk-production.csv', index_col = 'Month')
print(data.head())
print('\n')

#convert the index (month) to date time format to be used as time series
data.index = pd.to_datetime(data.index)

#print data info
print('Monthly Milk Production Info:')
print(data.info())
print('\n')

#print data description
print('Monthly Milk Production Description:')
print(data.describe())
print('\n')

#plot the time series data
print('Showing Time Series Data...')
data.plot()
plt.show()
print('\n')


"""
-----------------------------------------------------------------------------------------
Prepare the Data
-----------------------------------------------------------------------------------------
"""

#want to attempt to predict a year's worth of data (12 months, or steps, into the future)
#create a train test split using the head and tail for indexing

#the test set is the last 12 months of data, and everything before is training data
#168 rows, therefore training set = 168 - (12 months) = 156
train_set = data.head(156)
test_set = data.tail(12)

#scale the data using sk learn
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

#scale the training and testing sets 
scaled_train = scaler.fit_transform(train_set)
scaled_test = scaler.fit_transform(test_set)


"""
-----------------------------------------------------------------------------------------
Create a batch function to feed batches of the training data
-----------------------------------------------------------------------------------------
"""

def next_batch(training_data,batch_size,steps):
    """
    INPUT: Data, Batch Size, Time Steps per batch
    OUTPUT: A tuple of y time series results. y[:,:-1] and y[:,1:]
    """
    
    # Use np.random.randint to set a random starting point index for the batch.
    # Grab a random starting point for each batch
    rand_start = np.random.randint(0,len(training_data)-steps) 
    
    # Need to index the data from the random start to random start + steps
    # Reshape data to be (1,steps)
    # Create Y data for time series in batches
    y_batch = np.array(training_data[rand_start:rand_start+steps+1]).reshape(1,steps+1)

    # Return the batches. Two batches to return y[:,:-1] and y[:,1:]
    # Reshape into tensors for the RNN
    return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1) 

"""
-----------------------------------------------------------------------------------------
Set up the RNN Model
-----------------------------------------------------------------------------------------
"""

# One input: the time series
n_inputs = 1
# Number of steps per batch
n_time_steps = 12
# 100 neuron layer: adjust for different results
n_neurons = 100
# One output: the predicted time series
n_outputs = 1
# learning rate: adjust for different results
learning_rate = 0.001
# how many iterations (training steps): adjust for different results
n_train_iterations = 10000
# Size of the batch
batch_size = 1

#create placeholders
X = tf.placeholder(tf.float32, [None, n_time_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_time_steps, n_outputs])

#create the rnn layer, can choose OutputProjectionWrappers, BasicRNNCells, BasicLSTMCells, MultiRNNCell, GRUCell, etc...
#here, will use OutputProjectionWrapper around a BasicLSTMCell
cell = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons, activation=tf.nn.relu),
    output_size=n_outputs) 

#pass cell variable into model with placeholder x
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

#define loss function and optimizer
#will use mean squared error loss function, adamoptimizer
loss = tf.reduce_mean(tf.square(outputs - y)) # MSE
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

#initialize global variables
init = tf.global_variables_initializer()

#create saver instance
saver = tf.train.Saver()

#create and run session that trains on batches from next_batch function
#add loss evaluation every 100 steps
print("Showing Mean Squared Error Every 100 Training Iterations:")
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    
    for iteration in range(n_train_iterations):
        
        X_batch, y_batch = next_batch(scaled_train,batch_size,n_time_steps)
        sess.run(train, feed_dict={X: X_batch, y: y_batch})
        
        if iteration % 100 == 0:
            
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(iteration, "\tMSE:", mse)
    
    # Save Model for Later
    saver.save(sess, "./ex_time_series_model")


#print the test set dataframe
print('\nTest Set Contents:')
print(test_set)
print('\n')

#print its description
print('Test Set Description:')
print(test_set.describe())
print('\n')


#create and run session to generate 12 months of data based
#off the last 12 months of data in the training set 
with tf.Session() as sess:
    
    # Use your Saver instance to restore your saved rnn time series model
    saver.restore(sess, "./ex_time_series_model")

    # Create a numpy array for your genreative seed from the last 12 months of the 
    # training set data. Hint: Just use tail(12) and then pass it to an np.array
    train_seed = list(scaled_train[-12:])
    
    ## Now create a for loop that 
    for iteration in range(12):
        X_batch = np.array(train_seed[-n_time_steps:]).reshape(1, n_time_steps, 1)
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        train_seed.append(y_pred[0, -1, 0])

#get result of the predictions
print("Train Seed:")
print(train_seed)
print('\n')

#get results that are generated values, apply inverse transform to get value unites
#reshape to (12,1) to add to the test_set dataframe
generated_values = scaler.inverse_transform(np.array(train_seed[12:]).reshape(12,1))

#create new column in test_set for generated values
test_set['Generated Values'] = generated_values

#show the new test_set dataframe
print('\nTest Set Dataframe:')
print(test_set)
print('\n')

#plot the Milk Production data against the generated values
print("Showing Milk Production Data vs. Predicted Values...")
test_set.plot()
plt.title("Milk Production Data vs. Predicted Data")
plt.show()
print('\n')


""" 
Play around with the adjustable values in the RNN model setup section of this code.
Different values and combinations will greatly change the accuracy of the model.
How can these values be optimized?
"""


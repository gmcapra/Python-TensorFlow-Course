"""
-------------------------------------------------------------------------------------
Exploring Tensorflow Abstractions - Layers API
-------------------------------------------------------------------------------------
Gianluca Capraro
Created: July 2019
-----------------------------------------------------------------------------------------
The data used in this project is available to be imported from SciKit Learn and contains
data regarding Wine and its properties.

The purpose of this script is to examine the Tensorflow Layers API
offered and utilize it to classify wine into one of 3 possible labels.
-----------------------------------------------------------------------------------------
"""
#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#get the data
from sklearn.datasets import load_wine
data = load_wine()

#get the keys for navigating the dictionary returned
print("\nWine Data Dictionary Keys:")
print(data.keys())
print('\n')

#print the description provided by the dictionary
print("Wine Data Description:")
print(data.DESCR)
print('\n')

#define our feature data that will be used to make predictions
feature_data = data['data']

#define the target variables (what we want to predict)
targets = data['target']

#perform train test split on data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(feature_data, targets, test_size = 0.2)

#scale the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)

#get one hot encoded train and test sets
one_hot_y_train = pd.get_dummies(y_train).as_matrix()
one_hot_y_test = pd.get_dummies(y_test).as_matrix()

#set parameters for model
n_features = 13
n_hidden_1 = 13
n_hidden_2 = 13
n_outputs = 3
learning_rate = 0.01

#import fully connected from layers
from tensorflow.contrib.layers import fully_connected

#set placeholders
X = tf.placeholder(tf.float32,shape=[None,n_features])
y_true = tf.placeholder(tf.float32,shape=[None,3])

#define activation function
activation_func = tf.nn.relu

#create the model layers
hidden_1 = fully_connected(X,n_hidden_1,activation_fn=activation_func)
hidden_2 = fully_connected(hidden_1,n_hidden_2,activation_fn=activation_func)
output = fully_connected(hidden_2,n_outputs)

#define loss function
loss = tf.losses.softmax_cross_entropy(onehot_labels=y_true, logits=output)

#define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

#initialize variables
init = tf.global_variables_initializer()

#create and run tf session
training_steps = 1000
with tf.Session() as tf_sess:
    tf_sess.run(init)
    
    for i in range(training_steps):
        tf_sess.run(train,feed_dict={X:x_train_scaled,y_true:one_hot_y_train})
        
    # Get predictions from model
    logits = output.eval(feed_dict={X:x_test_scaled})
    predictions = tf.argmax(logits,axis=1)
    results = predictions.eval()

#get classification and confusion matrix reports from sklearn
from sklearn.metrics import confusion_matrix, classification_report
print("Classification Report for Wine Data Predictions Made with the Layer API:")
print(classification_report(results,y_test))
print('\n')
print("Confusion Matrix for Wine Data Predictions Made with the Layer API:")
print(confusion_matrix(results,y_test))
print('\n')


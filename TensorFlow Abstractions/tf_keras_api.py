"""
-------------------------------------------------------------------------------------
Exploring Tensorflow Abstractions - Keras
-------------------------------------------------------------------------------------
Gianluca Capraro
Created: July 2019
-----------------------------------------------------------------------------------------
The data used in this project is available to be imported from SciKit Learn and contains
data regarding Wine and its properties.

The purpose of this script is to examine the Keras API with TensorFlow and attempt
to classify wine into one of 3 target variables based on the features of the data.
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
X_train, X_test, y_train, y_test = train_test_split(feature_data, targets, test_size = 0.3)

#scale the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)

"""
-----------------------------------------------------------------------------------------
TensorFlow Keras
-----------------------------------------------------------------------------------------
"""

#import keras
from tensorflow.contrib.keras import models

#create the keras dnn model
keras_dnn = models.Sequential()

#add model layers
from tensorflow.contrib.keras import layers
keras_dnn.add(layers.Dense(units=13,input_dim=13,activation='relu'))
keras_dnn.add(layers.Dense(units=13,activation='relu'))
keras_dnn.add(layers.Dense(units=13,activation='relu'))
keras_dnn.add(layers.Dense(units=3,activation='softmax'))
keras_dnn.add(layers.Dense(units=3,activation='softmax'))

#compile the model
from tensorflow.contrib.keras import losses,optimizers,metrics
losses.sparse_categorical_crossentropy
keras_dnn.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#train the model
keras_dnn.fit(x_train_scaled,y_train,epochs=100)

#get predictions from the model
predictions = keras_dnn.predict_classes(x_test_scaled)

#get classification and confusion matrix reports from sklearn
from sklearn.metrics import classification_report, confusion_matrix
print("Classification Report for Wine Data Predictions Made with Keras:")
print(classification_report(y_test,predictions))
print('\n')
print("Confusion Matrix for Wine Data Predictions Made with Keras:")
print(confusion_matrix(y_test,predictions))
print('\n')











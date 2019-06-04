"""
---------------------------------------------------------------------------------------
Use TensorFlow Estimator API to Perform a Regression Task on a Dataset
---------------------------------------------------------------------------------------
Gianluca Capraro
Created: June 2019
---------------------------------------------------------------------------------------
The purpose of this script is to demonstrate the use of the TensorFlow
Estimator API to perform a Regression Task on an artificially created dataset.
---------------------------------------------------------------------------------------
"""
#import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

"""
---------------------------------------------------------------------------------------
Set up the Dataset
---------------------------------------------------------------------------------------
"""

#create large x dataset with 1 million points
x_data = np.linspace(0.0,10.0,1000000)

#define the noise for our dataset
noise = np.random.randn(len(x_data))

#define the bias
b = 5

#create y_true which holds the correct y values for our dataset
y_true =  (0.5 * x_data ) + 5 + noise

#combine x_data and y_true for a wholistic dataset
data = pd.concat([pd.DataFrame(data=x_data,columns=['X']),pd.DataFrame(data=y_true,columns=['Y'])],axis=1)

#visualize a sample set of the data (don't want to feed all of the 1 million points)
print('\nShowing 500 point Sample of Dataset:')
data_sample = data.sample(n = 500).plot(kind = 'scatter', x = 'X', y = 'Y')
plt.show()
print('\n')


"""
---------------------------------------------------------------------------------------
TensorFlow Estimator
---------------------------------------------------------------------------------------
"""

#define feature columns of the dataset, in this example we only have 1 (X)
feature_cols = [tf.feature_column.numeric_column('x', shape=[1])]

#define the estimator using the Linear Regressor model
estimator = tf.estimator.LinearRegressor(feature_columns = feature_cols)

#use sklearn to train test split the data
from sklearn.model_selection import train_test_split
x_train, x_eval, y_train, y_eval = train_test_split(x_data, y_true, test_size = 0.3)

#create the input functions (need two additional input funcs for training and evaluation)
#input function, batch size of 10, we will define the num steps later
input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=5,num_epochs=None,shuffle=True)
#training input func, no shuffle because want to verify results with eval
train_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train},y_train,batch_size=5,num_epochs=1000,shuffle=False)
#evaluation input func
eval_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_eval},y_eval,batch_size=5,num_epochs=1000,shuffle=False)

#train the estimator with the input function, define steps here
estimator.train(input_fn=input_func,steps=1000)

#evaluate the results of the input function
train_metrics = estimator.evaluate(input_fn=train_input_func,steps=1000)
eval_metrics = estimator.evaluate(input_fn=eval_input_func,steps=1000)

print("\nTrain metrics:")
print(train_metrics)
print('\n')
print("Eval metrics:")
print(eval_metrics)
print('\n')


"""
---------------------------------------------------------------------------------------
Getting Predictions From The Model
---------------------------------------------------------------------------------------
"""

#define input function to make predictions (predict 10 points to make a line of best fit)
input_fn_predict = tf.estimator.inputs.numpy_input_fn({'x':np.linspace(0,10,10)},shuffle=False)

#create numpy array for the individual predictions
predictions = []

#for each x value, make a prediction using the input function
for x in estimator.predict(input_fn=input_fn_predict):
    predictions.append(x['predictions'])

#plot the original sample dataset and the predicted data points
print('Showing 500 Point Data Sample and Predicted Linear Regression Line:')
data.sample(n=500).plot(kind='scatter',x='X',y='Y')
plt.plot(np.linspace(0,10,10),predictions,'r')
plt.show()
print('\n')


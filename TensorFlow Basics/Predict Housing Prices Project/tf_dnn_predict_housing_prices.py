"""
---------------------------------------------------------------------------------------
Creating a DNN Regressor Model to Predict Median House Values with TensorFlow
---------------------------------------------------------------------------------------
Gianluca Capraro
Created: June 2019
---------------------------------------------------------------------------------------
The purpose of this script is to predict the median house values of each block in
the California Housing Dataset. The California Housing Data contains information on
all of the block groups in California from the 1990 Census. To make the stated 
predictions, a Deep Neural Network Regressor Model will be created using the TensorFlow
API. This model will also be evaluated to determine its performance. 

The dataset can be found under California Housing, here:
http://www.dcc.fc.up.pt/~ltorgo/Regression/DataSets.html
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
Data Setup and Cleaning
---------------------------------------------------------------------------------------
"""
#read in the dataset and print the head
data = pd.read_csv('cal_housing_clean.csv')
print('\nCalifornia Housing (1990) Dataset Head:')
print(data.head())
print('\n')

#define the feature data that will be used to make predictions
x_data = data.drop(['medianHouseValue'], axis=1)

#define the column that is our target variable
y_labels = data['medianHouseValue']

#perform a train test split on the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_data, y_labels, test_size = 0.3)

#scale the feature data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

#fit the scaler to our feature training data only, we dont want to assume prior knowledge of testing data
scaler.fit(X_train)

#create two separate dataframes for scaled training and testing data
X_train = pd.DataFrame(data=scaler.transform(X_train),columns = X_train.columns,index=X_train.index)
X_test = pd.DataFrame(data=scaler.transform(X_test),columns = X_test.columns,index=X_test.index)

#define feature columns for tensorflow use
house_age = tf.feature_column.numeric_column('housingMedianAge')
n_rooms = tf.feature_column.numeric_column('totalRooms')
n_bedrooms = tf.feature_column.numeric_column('totalBedrooms')
population = tf.feature_column.numeric_column('population')
n_households = tf.feature_column.numeric_column('households')
income = tf.feature_column.numeric_column('medianIncome')

#define dataset of feature columns
features = [house_age,n_rooms,n_bedrooms,population,n_households,income]

#create input function for TF estimator object
input_func = tf.estimator.inputs.pandas_input_fn(
										x=X_train,
										y=y_train,
										batch_size=10,
										num_epochs=1000,
                                        shuffle=True)

#create the estimator DNN model, change the hidden units for different results
dnn_model = tf.estimator.DNNRegressor(hidden_units=[10,10,10],feature_columns=features)

#train the model for a certain number of steps (play around with the steps for different results)
dnn_model.train(input_fn=input_func,steps=15000)

#create the input function to obtain predictions
predict_input_func = tf.estimator.inputs.pandas_input_fn(
      									x=X_test,
      									batch_size=10,
      									num_epochs=1,
      									shuffle=False)

#get the predictions from the model
dnn_predictions = list(dnn_model.predict(predict_input_func))

#store only the median house price predictions in a new list
final_predictions = []
for prediction in dnn_predictions:
    final_predictions.append(prediction['predictions'])

"""
print('Median Housing Price Predictions:')
print(final_predictions)
print('\n')
"""

#determine the mean squared error between the known and predicted median housing prices
from sklearn.metrics import mean_squared_error
print('\nRoot Mean Squared Error (RMSE) - Median Housing Price Predictions:')
print(mean_squared_error(y_test,final_predictions)**0.5)
print('\n')


"""
The RMSE we calculate for our predictions is not that great, considering the median housing
price is around $200,000. Try to improve this prediction accuracy by using a different model,
or changing the parameters we used above.
"""
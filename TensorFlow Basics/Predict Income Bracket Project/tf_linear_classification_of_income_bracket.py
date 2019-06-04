"""
---------------------------------------------------------------------------------------
Build a Linear Classification Model with TensorFlow to Predict Income Bracket
---------------------------------------------------------------------------------------
Gianluca Capraro
Created: June 2019
---------------------------------------------------------------------------------------
The purpose of this script is to demonstrate the use of the TensorFlow Estimator API
to build a Linear Classification model that will be used to predict a person's income
bracket based on available features. For this project, the California Census data .csv
file will be used. The .csv file can be found in the same folder as this script.
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
data = pd.read_csv('census_data.csv')
print('\nCalifornia Census Dataset Head:')
print(data.head())
print('\n')


#
data['income_bracket'].unique()

#
def correct_labels(label):
    if label==' <=50K':
        return 0
    else:
        return 1

#
data['income_bracket'] = data['income_bracket'].apply(correct_labels)

"""
---------------------------------------------------------------------------------------
Train Test Split
---------------------------------------------------------------------------------------
"""

#import the train test split from sklearn
from sklearn.model_selection import train_test_split

#separate our data into features and the target class
x_data = data.drop('income_bracket',axis=1)
y_labels = data['income_bracket']
 
#split the data into testing and training sets
X_train, X_test, y_train, y_test = train_test_split(x_data,y_labels,test_size=0.3)

"""
---------------------------------------------------------------------------------------
Build the Model with TensorFlow
---------------------------------------------------------------------------------------
"""

#define our categorical features using either a vocab list or hash bucketing
gender = tf.feature_column.categorical_column_with_vocabulary_list("gender", ["Female", "Male"])
occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation", hash_bucket_size=1000)
marital_status = tf.feature_column.categorical_column_with_hash_bucket("marital_status", hash_bucket_size=1000)
relationship = tf.feature_column.categorical_column_with_hash_bucket("relationship", hash_bucket_size=1000)
education = tf.feature_column.categorical_column_with_hash_bucket("education", hash_bucket_size=1000)
workclass = tf.feature_column.categorical_column_with_hash_bucket("workclass", hash_bucket_size=1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket("native_country", hash_bucket_size=1000)

#define the numerical features for use in tensorflow
age = tf.feature_column.numeric_column("age")
education_num = tf.feature_column.numeric_column("education_num")
capital_gain = tf.feature_column.numeric_column("capital_gain")
capital_loss = tf.feature_column.numeric_column("capital_loss")
hours_per_week = tf.feature_column.numeric_column("hours_per_week")

#put all our features together into one dataset for tensorflow estimator
features = [gender,occupation,marital_status,relationship,education,workclass,native_country,
            age,education_num,capital_gain,capital_loss,hours_per_week]

#create the input function with batchsize of 100, shuffle for training input function
input_func = tf.estimator.inputs.pandas_input_fn(
									x=X_train,
									y=y_train,
									batch_size=100,
									num_epochs=None,
									shuffle=True)

#create the linear classifier model
model = tf.estimator.LinearClassifier(feature_columns=features)

#train the model using the input function with 10,000 steps
model.train(input_fn=input_func,steps=10000)

#create the prediction input function using the X testing set
prediction_func = tf.estimator.inputs.pandas_input_fn(
									x=X_test,
									batch_size=len(X_test),
									shuffle=False)

#get the predictions dictionary from the model
predictions = list(model.predict(input_fn=prediction_func))

#store only the final prediction values in a new list
final_predictions = [prediction['class_ids'][0] for prediction in predictions]

"""
---------------------------------------------------------------------------------------
Evaluate the Model with SKLearn
---------------------------------------------------------------------------------------
"""

#import classification report and confusion matrix from sklearn
from sklearn.metrics import classification_report,confusion_matrix

#show the results of our models predictive performance, how did the model perform?
#how can the model be improved?
#can a DNN Classifier be used instead of the Linear Classifier? need to change feature columns to embedded features
print('Classification Report for Income Bracket Predictions:')
print(classification_report(y_test,final_predictions))
print('\n')

print('Confusion Matrix for Income Bracket Predictions:')
print(confusion_matrix(y_test,final_predictions))
print('\n')



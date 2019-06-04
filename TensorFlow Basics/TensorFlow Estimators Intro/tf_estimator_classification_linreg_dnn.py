"""
---------------------------------------------------------------------------------------
Use TensorFlow Estimator API to Perform Binary Classification on a Dataset
---------------------------------------------------------------------------------------
Gianluca Capraro
Created: June 2019
---------------------------------------------------------------------------------------
The purpose of this script is to demonstrate the use of the TensorFlow
Estimator API to perform a Classification Task on the Pima Indians Diabetes Dataset.
In this example, we are performing Binary classification to determine if an individual
would fall into class 0 (no diabetes) or class 1 (has diabetes) based on the numerical
and categorical features of the dataset. This script will show how to use a Linear 
Regression Classification Model, as well as a Deep Neural Network Classifier to make
predictions.

The Pima Indians Diabetes Database can be found here:
https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes
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
pima_data = pd.read_csv('pima-indians-diabetes.csv')
print('\nPima Indians Diabetes Dataset Head:')
print(pima_data.head())
print('\n')

#define the numerical columns to normalize
#class is removed because we are trying to predict this label
#group and age are removed as they are or will be used as categorical columns
cols_to_norm = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps',
       'Insulin', 'BMI', 'Pedigree']

#normalize the columns
pima_data[cols_to_norm] = pima_data[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

#print the normalized data head
print('Normalized Data:')
print(pima_data.head())
print('\n')

#define the continuous features for use in tensorflow
num_preg = tf.feature_column.numeric_column('Number_pregnant')
plasma_gluc = tf.feature_column.numeric_column('Glucose_concentration')
dias_press = tf.feature_column.numeric_column('Blood_pressure')
tricep = tf.feature_column.numeric_column('Triceps')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
diabetes_pedigree = tf.feature_column.numeric_column('Pedigree')
age = tf.feature_column.numeric_column('Age')

#get the categorical feature for the assigned group (a,b,c,d)
#we know there are only 4 possible groups so we can use vocab list
assigned_group = tf.feature_column.categorical_column_with_vocabulary_list('Group',['A','B','C','D'])
#if we wanted to automatically bucket for groups, either because there are too many to type
#or if they are not all known, the hashbucket command can be used (pass in max # of groups you expect as the hash bucket size)
# assigned_group = tf.feature_column.categorical_column_with_hash_bucket('Group', hash_bucket_size=10)

#feature engineering (convert continuous age feature to a categorical feature)
#instead of individual ages, we want to only keep age buckets as a feature
age_buckets = tf.feature_column.bucketized_column(age, boundaries=[20,30,40,50,60,70,80])

#now we can redefine the feature columns with our assigned groups and bucketed ages
feat_cols = [num_preg,plasma_gluc,dias_press,tricep,insulin,bmi,diabetes_pedigree,assigned_group,age_buckets]


"""
---------------------------------------------------------------------------------------
Train Test Split
---------------------------------------------------------------------------------------
"""

#define what the x and y data will be from our dataset
x_data = pima_data.drop('Class',axis=1)
y_labels = pima_data['Class']


#import and perform the split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_data, y_labels, test_size = 0.3)


"""
---------------------------------------------------------------------------------------
Creating a Linear Classification Model Using TF Estimator
---------------------------------------------------------------------------------------
"""

#define our input function using a batch size of 10 and 1000 steps
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=1000,shuffle=True)

#create the model specifying the classes and features
model = tf.estimator.LinearClassifier(feature_columns=feat_cols, n_classes=2)

#train the model
model.train(input_fn=input_func,steps=1000)


"""
---------------------------------------------------------------------------------------
Evaluating the Linear Classification Model
---------------------------------------------------------------------------------------
"""

#define the evaluation input function
eval_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_test,
      y=y_test,
      batch_size=10,
      num_epochs=1,
      shuffle=False)


#print the accuracy results for the model
linreg_results = model.evaluate(eval_input_func)
print('\nLinear Regression Model - Evaluation Metrics for Diabetes Classification:')
print(linreg_results)
print('\n')


"""
---------------------------------------------------------------------------------------
Making Predictions for Diabetes Classification from the Linear Classification Model
---------------------------------------------------------------------------------------
"""

#define the prediction input function using our test data
pred_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_test,
      batch_size=10,
      num_epochs=1,
      shuffle=False)


#Get individual predictions from input function
#this returns a list of the predictions made and the probabilities for belonging to each class
linreg_predictions = list(model.predict(pred_input_func))

#print the predictions (long list)
#print('Showing Linear Regression Classifier Predictions based on X_Test Data:')
#print(linreg_predictions)


"""
---------------------------------------------------------------------------------------
Creating a Deep Neural Network Classifier Using TF Estimator
---------------------------------------------------------------------------------------
"""

#need to pass feature column and number of groups for assigned groups into embedded column 
embedded_group_column = tf.feature_column.embedding_column(assigned_group, dimension=4)

#now, reset our feature columns to include the embedded group
feat_cols = [num_preg,plasma_gluc,dias_press,tricep,insulin,bmi,diabetes_pedigree,embedded_group_column,age_buckets]

#redefine the input function for the DNN model
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=1000,shuffle=True)

#create the DNN model using TF estimator
dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,10,10],feature_columns=feat_cols,n_classes=2)

#train the model over 1000 steps
dnn_model.train(input_fn=input_func,steps=1000)

#create the evaluation input function using our test data
eval_input_func = tf.estimator.inputs.pandas_input_fn(
      x=X_test,
      y=y_test,
      batch_size=10,
      num_epochs=1,
      shuffle=False)

#obtain the results from the evaluation of our dnn model
dnn_results = dnn_model.evaluate(eval_input_func)

#print the results -- how did the DNN classifier perform compared to the linreg model?
#what happens when we add more hidden layers or neurons to the DNN? does the accuracy improve or are we just overfitting?
print('\nDNN Model - Evaluation Metrics for Diabetes Classification:')
print(dnn_results)
print('\n')


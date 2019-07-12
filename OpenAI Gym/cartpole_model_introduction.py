"""
-------------------------------------------------------------------------------------
Simple Neural Network Game Introduction - CartPole Problem
-------------------------------------------------------------------------------------
Gianluca Capraro
Created: July 2019
-----------------------------------------------------------------------------------------
The purpose of this script is to create and run a simple game for the cart pole problem
that attempts to stay balanced for as many steps as possible. This script is left open to 
be revised and be reconfigured to have a higher score.
-----------------------------------------------------------------------------------------
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
print("\nGym Library Found.\n")

#create network variables
n_inputs = 4
n_hidden = 4
n_outputs = 1 #probability to go left

#create initializer that is scaled
initializer = tf.contrib.layers.variance_scaling_initializer()

#placeholder values
X = tf.placeholder(tf.float32,shape=[None,n_inputs])

#create layers
hidden_1 = tf.layers.dense(X,n_hidden,activation=tf.nn.relu,kernel_initializer = initializer)
hidden_2 = tf.layers.dense(hidden_1,n_hidden,activation=tf.nn.relu,kernel_initializer = initializer)
output_layer = tf.layers.dense(hidden_2,n_outputs,activation=tf.nn.relu,kernel_initializer = initializer)

#create probabilities
probabilities = tf.concat(axis = 1,values = [output_layer, 1-output_layer])

#create actions
action = tf.multinomial(probabilities,num_samples=1)

init = tf.global_variables_initializer()

#create and run session for cart pole environment
epi = 250
step_limit = 500
env = gym.make('CartPole-v0')
avg_steps = []
with tf.Session() as tf_sess:
	
	init.run()

	for x_episode in range(epi):
		observation = env.reset()

		for step in range(step_limit):
			action_value = action.eval(feed_dict={X:observation.reshape(1,n_inputs)})
			observation,reward,done,info = env.step(action_value[0][0]) #returns 0 or 1

			if done:
				avg_steps.append(step)
				print("Done after {} steps.".format(step))
				break

print("After {} episodes, average steps per game was {}\n".format(epi,np.mean(avg_steps)))
env.close()


"""
As can be observed, the model does not perform particularly well. This is due to a number of
reasons including that we are not using a history to build on. This is a good place to 
build on. In the next project, will revisit with Policy Gradients to train with reinforcement learning.
"""


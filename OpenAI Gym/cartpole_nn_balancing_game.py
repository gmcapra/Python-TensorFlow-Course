"""
-------------------------------------------------------------------------------------
Policy Gradients & Playing a Neural Network CartPole Game
-------------------------------------------------------------------------------------
Gianluca Capraro
Created: July 2019
-----------------------------------------------------------------------------------------
The purpose of this script is to build upon the simple balancing game previously created
using the cart pole model and teach the cart to balance itself and keep the pole upright.

This model can be modified to not only keep the pole balanced, but also factor in keeping
the cart on the screen for as long as possible. (Something to try)
-----------------------------------------------------------------------------------------
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
print("\nGym Library Found.\n")

#define model parameters
n_inputs = 4
n_hidden = 4
n_outputs = 1
learning_rate = 0.01

#create initializer, scaled
initializer = tf.contrib.layers.variance_scaling_initializer()

#define placeholder value
X = tf.placeholder(tf.float32,shape=[None,n_inputs])

#create hidden layer
hidden_layer = tf.layers.dense(X,n_hidden,activation=tf.nn.elu,kernel_initializer=initializer)

#define logits and outputs from logits
logits = tf.layers.dense(hidden_layer,n_outputs)
outputs = tf.nn.sigmoid(logits)

#define probabilities and resulting action
probabilities = tf.concat(axis=1,values = [outputs,1-outputs])
action = tf.multinomial(probabilities,num_samples=1)
y = 1.0 - tf.to_float(action)

#define loss function and optimizer
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate)
gradients_variables = optimizer.compute_gradients(cross_entropy)
gradients = []
gradient_placeholders = []
grads_and_vars_feed = []
for gradient, variable in gradients_variables:
    gradients.append(gradient)
    gradient_placeholder = tf.placeholder(tf.float32, shape=gradient.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder, variable))

training_op = optimizer.apply_gradients(grads_and_vars_feed)

"""
-----------------------------------------------------------------------------------------
Create Rewards Functions
-----------------------------------------------------------------------------------------
https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-2-ded33892c724
"""

def helper_discount_rewards(rewards, discount_rate):
    #takes in rewards, applies discount
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

def discount_and_normalize_rewards(all_rewards, discount_rate):
    #takes in all rewards, performs helper discount function and normalizes
    all_discounted_rewards = []
    for rewards in all_rewards:
        all_discounted_rewards.append(helper_discount_rewards(rewards,discount_rate))

    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]

"""
-----------------------------------------------------------------------------------------
Training Session
-----------------------------------------------------------------------------------------
"""

init = tf.global_variables_initializer()
saver = tf.train.Saver()

#create cartpole environment
env = gym.make("CartPole-v0")
#define session parameters
n_game_rounds = 10
max_steps = 1000
n_iterations = 250
discount_rate = 0.95

#run session with parameters
with tf.Session() as tf_sess:
    tf_sess.run(init)

    for iteration in range(n_iterations):
        print("Current Training Iteration: {} \n".format(iteration) )

        all_rewards = []
        all_gradients = []

        for game in range(n_game_rounds):

            current_rewards = []
            current_gradients = []
            observations = env.reset()

            for step in range(max_steps):
                action_val, gradients_val = tf_sess.run([action, gradients], feed_dict={X: observations.reshape(1, n_inputs)})
                # Perform Action
                observations, reward, done, info = env.step(action_val[0][0])
                # Get Current Rewards and Gradients
                current_rewards.append(reward)
                current_gradients.append(gradients_val)
                if done:
                    # Game Ended
                    break

            # Append to list of all rewards
            all_rewards.append(current_rewards)
            all_gradients.append(current_gradients)

        all_rewards = discount_and_normalize_rewards(all_rewards,discount_rate)
        feed_dict = {}

        for var_index, gradient_placeholder in enumerate(gradient_placeholders):
            mean_gradients = np.mean([reward * all_gradients[game_index][step][var_index]
                                      for game_index, rewards in enumerate(all_rewards)
                                          for step, reward in enumerate(rewards)], axis=0)
            feed_dict[gradient_placeholder] = mean_gradients

        tf_sess.run(training_op, feed_dict=feed_dict)

    print('Saving graph and session')
    meta_graph_def = tf.train.export_meta_graph(filename='models/my-policy-model')
    saver.save(tf_sess, 'models/my-policy-model')

"""
-----------------------------------------------------------------------------------------
Run Trained Model
-----------------------------------------------------------------------------------------
"""
env = gym.make('CartPole-v0')

observations = env.reset()
with tf.Session() as tf_sess:
    # https://www.tensorflow.org/api_guides/python/meta_graph
    new_saver = tf.train.import_meta_graph('models/my-policy-model.meta')
    new_saver.restore(tf_sess,'models/my-policy-model')

    for x in range(500):
        env.render()
        action_val, gradients_val = tf_sess.run([action, gradients], feed_dict={X: observations.reshape(1, n_inputs)})
        observations, reward, done, info = env.step(action_val[0][0])

print("\nPole Fell Over.\nI'm sorry. I tried my best.\n")

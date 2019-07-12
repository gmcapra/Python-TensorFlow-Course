"""
-------------------------------------------------------------------------------------
Intro to OpenAI Gym
-------------------------------------------------------------------------------------
Gianluca Capraro
Created: July 2019
-----------------------------------------------------------------------------------------
The purpose of this script is to demonstrate how to create and modify a simple OpenAI Gym
environment and review some basic concepts on its use and documentation.
-----------------------------------------------------------------------------------------
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
print("\nGym Library Found.\n")

"""
-----------------------------------------------------------------------------------------
Gym Environment Basics - The Cart Pole Problem

-There is a pole on a cart. The cart can move left or right, and as a result the
pole will end up sliding in different directions.

-Actions to move cart and balance out the pole (0:Left, 1:Right)
-Environment is numpy array with 4 numbers (x position, x velocity, pole angle, angular velocity)
-These numbers are used as agent
-----------------------------------------------------------------------------------------
"""

#create the cartpole environment from open ai gym
#here is where you can play around with different available environments
env = gym.make('CartPole-v0')
env.reset()

"""
-----------------------------------------------------------------------------------------
Run the CartModel, Untrained
-----------------------------------------------------------------------------------------
n_steps = 1000
for _ in range(n_steps):
	env.render()
	env.step(env.action_space.sample()) #provides a random action sample
-----------------------------------------------------------------------------------------
"""

"""
-----------------------------------------------------------------------------------------
Gym Observations

-Environment step() function returns 
-Observation: env specific information that represents observations (angles, velocities, game states, etc.)
-Reward: amount of reward achieved by previous action (agent usually wants to increase reward)
-Done: boolean indicating if the environment needs to be reset (pole falls over, steps completed)
-Info: dictionary object with diagnostic info for debugging
-----------------------------------------------------------------------------------------
"""

#for a certain number of steps create a random action
#return the observation, reward, done, and info that is returned
#Print some observations
print("First Observations:")
for _ in range(1):

	action = env.action_space.sample()
	observation,reward,done,info = env.step(action)

	print("\nRandom Action Performed\n")
	print('Observation:')
	print(observation)
	print('Reward:')
	print(reward)
	print('Done:')
	print(done)
	print('Info:')
	print(info)
	print('\n')


"""
-----------------------------------------------------------------------------------------
Gym Actions

-Here we will implement actions to take based on only angle of the pole
-Policy, if the pole falls to the right, move the cart to the right, and vice versa.
-This is left as a barebones framework to be further refined, how long can you keep the pole up with your own conditions?
-----------------------------------------------------------------------------------------
"""

n_steps = 300
for _ in range(300):

	env.render()
	#grab what is returned from observation
	x_pos, x_vel, pole_ang, ang_vel = observation

	#learning agent has some check for an action
	if pole_ang > 0:
		action = 1
	else:
		action = 0

	#returns a new set
	observation,reward,done,info = env.step(action)



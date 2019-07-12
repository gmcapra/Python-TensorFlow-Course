"""
-------------------------------------------------------------------------------------
Generative Adversarial Networks (GAN) Project - Generate MNIST Handwritten Numbers
-------------------------------------------------------------------------------------
Gianluca Capraro
Created: July 2019
-----------------------------------------------------------------------------------------
A generative adversarial network or GAN is a type of machine learning system invented by 
Ian Goodfellow and co in 2014. The idea is that two neural networks contest with each 
other in a game. Given a training set, the model learns to generate new data with the 
same statistics as the training set.

The purpose of this script is to explore how Tensorflow can be used to create a GAN 
that is able to generate numbers based off of the MNIST data set.
-----------------------------------------------------------------------------------------
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../03-Convolutional-Neural-Networks/MNIST_data/",one_hot=True)

plt.imshow(mnist.train.images[5].reshape(28,28),cmap='Greys')

"""
The Generator
"""
def generator(z,reuse=None):
    with tf.variable_scope('gen',reuse=reuse):
        hidden1 = tf.layers.dense(inputs=z,units=128)
        alpha = 0.01
        hidden1 = tf.maximum(alpha*hidden1,hidden1)
        hidden2 = tf.layers.dense(inputs=hidden1,units=128)
       
        hidden2 = tf.maximum(alpha*hidden2,hidden2)
        output = tf.layers.dense(hidden2,units=784,activation=tf.nn.tanh)
        return output
    
"""
The Discriminator
"""
def discriminator(X,reuse=None):
    with tf.variable_scope('dis',reuse=reuse):
        hidden1 = tf.layers.dense(inputs=X,units=128)
        alpha = 0.01
        hidden1 = tf.maximum(alpha*hidden1,hidden1)
        hidden2 = tf.layers.dense(inputs=hidden1,units=128)
        hidden2 = tf.maximum(alpha*hidden2,hidden2)
        logits = tf.layers.dense(hidden2,units=1)
        output = tf.sigmoid(logits)
    
        return output, logits


#define placeholders
real_images = tf.placeholder(tf.float32,shape=[None,784])
z = tf.placeholder(tf.float32,shape=[None,100])

#create generator
G = generator(z)

#create discriminators
D_out_real, D_logits_real = discriminator(real_images)
D_out_fake, D_logits_fake = discriminator(G,reuse=True)

#define losses
def loss_func(logits_in,labels_in):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_in,labels=labels_in))

D_real_loss = loss_func(D_logits_real,tf.ones_like(D_logits_real)* (0.9))
D_fake_loss = loss_func(D_logits_fake,tf.zeros_like(D_logits_real))
D_loss = D_real_loss + D_fake_loss

G_loss = loss_func(D_logits_fake,tf.ones_like(D_logits_fake))

#define optimizers
learning_rate = 0.001
tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'dis' in var.name]
g_vars = [var for var in tvars if 'gen' in var.name]
print([v.name for v in d_vars])
print([v.name for v in g_vars])
D_trainer = tf.train.AdamOptimizer(learning_rate).minimize(D_loss, var_list=d_vars)
G_trainer = tf.train.AdamOptimizer(learning_rate).minimize(G_loss, var_list=g_vars)

"""
The Training Session
"""
#define model parameters
batch_size = 100
epochs = 250 #will take a very long time, should be > 100 for good results
init = tf.global_variables_initializer()
saver = tf.train.Saver(var_list=g_vars)
samples = []

#create and run session
with tf.Session() as tf_sess:
    
    tf_sess.run(init)
    
    for e in range(epochs):

        num_batches = mnist.train.num_examples // batch_size
        
        for i in range(num_batches):
            
            #grab batch of images
            batch = mnist.train.next_batch(batch_size)
            
            #get images then reshape and rescale to pass to D
            batch_images = batch[0].reshape((batch_size, 784))
            batch_images = batch_images*2 - 1
            
            #random latent noise data for Generator
            # -1 to 1 - tanh activation
            batch_z = np.random.uniform(-1, 1, size=(batch_size, 100))
            
            #run optimizers
            _ = tf_sess.run(D_trainer, feed_dict={real_images: batch_images, z: batch_z})
            _ = tf_sess.run(G_trainer, feed_dict={z: batch_z})
        
            
        print("Currently on Epoch {} of {}".format(e+1, epochs))
        
        # Sample from generator, training for viewing afterwards
        sample_z = np.random.uniform(-1, 1, size=(1, 100))
        gen_sample = tf_sess.run(generator(z ,reuse=True),feed_dict={z: sample_z})
        samples.append(gen_sample)
        saver.save(tf_sess, 'models/epoch_model.ckpt')
   
"""
Generating New Samples
"""     
saver = tf.train.Saver(var_list=g_vars)
new_samples = []
#run session to generate samples
with tf.Session() as tf_sess:
    
    saver.restore(tf_sess,'models/epoch_model.ckpt')
    
    for x in range(5):
        sample_z = np.random.uniform(-1,1,size=(1,100))
        gen_sample = tf_sess.run(generator(z,reuse=True),feed_dict={z:sample_z})
        
        new_samples.append(gen_sample)

#show some of the generated samples
print("Showing Samples Generated from the MNIST Dataset...")
print('\n')
plt.imshow(samples[0].reshape(28,28),cmap='Greys')
plt.show()
plt.imshow(samples[1].reshape(28,28),cmap='Greys')
plt.show()
plt.imshow(samples[2].reshape(28,28),cmap='Greys')
plt.show()

"""
Note: For good generated number results, use an epoch greater than 100 or 500.
The model needs to be adequately trained on the MNIST dataset to be able to
generate accurate images.
"""


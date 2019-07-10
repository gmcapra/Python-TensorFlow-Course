"""
-------------------------------------------------------------------------------------
Word2Vec Tutorial Project - Using Word2Vec with Tensorflow and Python
-------------------------------------------------------------------------------------
Gianluca Capraro
Created: July 2019
-----------------------------------------------------------------------------------------
The goal of the Word2Vec model is learning word embeddings by modeling every word as a 
vector in n-D space. With words represented as vectors, we can easily check their 
similarity, add/subtract them, and perform other vector operations.
-----------------------------------------------------------------------------------------
"""

"""
-----------------------------------------------------------------------------------------
Import the libraries we will need
-----------------------------------------------------------------------------------------
"""
import math
import random
import zipfile
import os
import collections
import errno
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
from six.moves import urllib
from six.moves import xrange 

"""
-----------------------------------------------------------------------------------------
Get the Data
-----------------------------------------------------------------------------------------
"""
#define where the word data is located
data_dir = "word2vec_data/words"
data_url = 'http://mattmahoney.net/dc/text8.zip'

#download the data if it doesnt exist, open if it does
def fetch_words_data(url=data_url, words_data=data_dir):
    
    # Make the Dir if it does not exist
    os.makedirs(words_data, exist_ok=True)
    
    # Path to zip file 
    zip_path = os.path.join(words_data, "words.zip")
    
    # If the zip file isn't there, download it from the data url
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(url, zip_path)
        
    # Now that the zip file is there, get the data from it
    with zipfile.ZipFile(zip_path) as f:
        data = f.read(f.namelist()[0])
    
    # Return a list of all the words in the data source.
    return data.decode("ascii").split()


#get list of all words in data source
words = fetch_words_data()

#get length
print("\nLength of words")
print(len(words))
print('\n')

#print out a slice of words
print("Slice of words from 1050 to 1100")
print(words[1050:1100])
print('\n')

#reformat for clarity
for word in words[1050:1100]:
    print(word,end=' ')
print('\n')


"""
-----------------------------------------------------------------------------------------
Create Word Data and Vocabulary
-----------------------------------------------------------------------------------------
"""

#import the counter
from collections import Counter

def create_counts(vocab_size=50000):

    # Begin adding to vocab count with Counter
    vocab = [] + Counter(words).most_common(vocab_size)
    
    # Turn into a numpy array
    vocab = np.array([word for word, _ in vocab])
    
    dictionary = {word: code for code, word in enumerate(vocab)}
    data = np.array([dictionary.get(word, 0) for word in words])
    return data,vocab

#set the size of vocabulary
vocab_size = 50000
data, vocabulary = create_counts(vocab_size=vocab_size)

#what is the word at index 99, and what is the value of data at index 99?
print("Word and Data at Index 99:")
print(words[99],data[99])
print("\n")


"""
-----------------------------------------------------------------------------------------
Function for Batch Generation
-----------------------------------------------------------------------------------------
"""

def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
    if data_index == len(data):
        buffer[:] = data[:span]
        data_index = span
    else:
        buffer.append(data[data_index])
        data_index += 1

    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

"""
-----------------------------------------------------------------------------------------
Create Constants
-----------------------------------------------------------------------------------------
"""
#initialize data_index
data_index=0

#get some example batch with labels
batch, labels = generate_batch(8, 2, 1)

#define batch size
batch_size = 128

#define embedding vector dimension
embedding_size = 150

#how many words to consider on left and right
skip_window = 1       

#how many times to reuse input
num_skips = 2        

#random set of words to evaluate similarity on
valid_size = 16   

#pick samples in the head of the distribution
valid_window = 100  
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

#number of negative examples to sample
num_sampled = 64   

#set the learning rate
learning_rate = 0.01

# How many words in vocabulary
vocabulary_size = 50000

"""
-----------------------------------------------------------------------------------------
Create Tensorflow Placeholders and Constants
-----------------------------------------------------------------------------------------
"""

#reset graph
tf.reset_default_graph()

#define input placeholders and constant
train_inputs = tf.placeholder(tf.int32, shape=[None])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

#look up embeddings for inputs.
init_embeds = tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
embeddings = tf.Variable(init_embeds)
embed = tf.nn.embedding_lookup(embeddings, train_inputs)

#build variables for NCE loss fybctuib
nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],stddev=1.0 / np.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

#compute average loss for the batch
loss = tf.reduce_mean(
    tf.nn.nce_loss(nce_weights, nce_biases, train_labels, embed,
                   num_sampled, vocabulary_size))

#construct the Adam optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=1.0)
trainer = optimizer.minimize(loss)

#compute cosine similarity between minibatch examples and embeddings
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), axis=1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

# define global variable initializer
init = tf.global_variables_initializer()

#create and run the session with a certain number of steps
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

#set this to test (might take a while, can make larger later)
num_steps = 5000

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    average_loss = 0
    for step in range(num_steps):
         
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

        # We perform one update step by evaluating the training op (including it
        # in the list of returned values for session.run()
        empty, loss_val = sess.run([trainer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 1000 == 0:
            if step > 0:
                average_loss /= 1000
            # The average loss is an estimate of the loss over the last 1000 batches.
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0

    final_embeddings = normalized_embeddings.eval()

print('\n')


"""
------------------------------------------------------------------------------
Visualize the results
------------------------------------------------------------------------------
"""

#define plotting function
def plot_with_labels(low_dim_embs, labels):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  #in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')


#import sklearn
from  sklearn.manifold import TSNE
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)

#define number to plot
n_to_plot = 2000
low_dim_embs = tsne.fit_transform(final_embeddings[:n_to_plot,:])

#define labels
labels = [vocabulary[i] for i in range(n_to_plot)]

#call the plot function for number of labels to plot
print("Showing full words plot...")
plot_with_labels(low_dim_embs, labels)
plt.show()
print('\n')
#call the plot function for a smaller section
print("Showing small section of plot...")
plot_with_labels(low_dim_embs, labels)
plt.xlim(-5,5)
plt.ylim(-5,5)
plt.show()
print('\n')

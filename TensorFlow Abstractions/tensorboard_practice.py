"""
-------------------------------------------------------------------------------------
TensorBoard Practice Exercise
-------------------------------------------------------------------------------------
Gianluca Capraro
Created: July 2019
-----------------------------------------------------------------------------------------
The purpose of this script is to demonstrate how TensorBoard can be used with TensorFlow
and Python to help visualize graphs and operations that are run within a session.
-----------------------------------------------------------------------------------------
"""
import tensorflow as tf

#define some operations
with tf.name_scope("A_Operations"):
	a = tf.add(5,10,name="First_Add")
	a1 = tf.add(100,200,name="A_Add")
	a2 = tf.multiply(a,a1)

with tf.name_scope("B_Operations"):
	b = tf.add(15,20,name="Second_Add")
	b1 = tf.add(300,400,name="B_Add")
	b2 = tf.multiply(b,b1)

c = tf.multiply(a2,b2,name="Final_Result")

#create and run session
with tf.Session() as tf_sess:
	file_writer = tf.summary.FileWriter("./output",tf_sess.graph)
	print(tf_sess.run(c))
	file_writer.close()

"""
After running, can go to Terminal and type tensorboard --logdir="./output"
Then, copy the address that is displayed (this will link to TensorBoard)
Finally, paste the address in your browser and view visualization
"""


















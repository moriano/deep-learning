import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

"""
This is an example of a neuronal network with two hidden layers
All done usuing the low level api of tensorflow
"""

"""
Explore the dataset
Note that the X_train/y_train are not to be used directly in tensorflow, 
except for evaluation purposes, this is here solely so one can easily look 
at the dimension of the tensors
"""
X_train = mnist.train.images
y_train = mnist.train.labels
print("X_train shape is ", X_train.shape)
print("y_train shape is ", y_train.shape)

X_validation = mnist.validation.images
y_validation = mnist.validation.labels
print("X_validation shape is ", X_validation.shape)
print("y_validation shape is ", y_validation.shape)

X_test = mnist.test.images
y_test = mnist.test.labels
print("X_test shape is ", X_test.shape)
print("y_test shape is ", y_test.shape)


features_size = 28*28
labels_size = 10
"""
We need to define placeholders for the X and y values, as those values do NOT 
change, tf.placeholder() is the correct approach to go. The reason for shape 
to be None, someNumber is that we do not know how many rows we will be using, 
as that will depend on the size of the batches, so None will allow us to 
set a variable number of rows
"""
X_holder = tf.placeholder(tf.float32, shape=(None, features_size))
y_holder = tf.placeholder(tf.float32, shape=(None, labels_size))

"""
Finally we reach weights and bias terms, as these values will change, we need 
to store them as tf.Variable()
"""

"""
VERY IMPORTANT NOTE ON WEIGHTS.

The weight initialisation is extremely important, normally we want values to be 
close to each other. 

The default standard deviation when using tf.truncated_normal() is 1.0 and 
that is not good.

As a general rule, it is good to use a value equal to 

1/sqrt(n)

Where n is the number of neurons

In this case this would be 

1/sqrt(256) = 0.0625

In this example I just used 0.1 and that yields results of 
0.98 / 0.97 accuracy in train / test sets.

Using the default stdev = 1.0 the accuracy results are
0.938 / 0.90 in train / test 

So remember, weight initialization is critical!
"""
weights_hidden_1 = tf.Variable(tf.truncated_normal((features_size, 256), stddev=0.1))
bias_hidden_1 = tf.Variable(tf.zeros(256))

weights_hidden_2 = tf.Variable(tf.truncated_normal((256, 128), stddev=0.1))
bias_hidden_2 = tf.Variable(tf.zeros(128))

weights_output = tf.Variable(tf.truncated_normal((128, labels_size), stddev=0.1))
bias_output = tf.Variable(tf.zeros(labels_size))

# Define the predictions, simply do matrix multiplication and add the bias
hidden_input_1 = tf.add(tf.matmul(X_holder, weights_hidden_1), bias_hidden_1)
hidden_output_1 = tf.nn.relu(hidden_input_1)

hidden_input_2 = tf.add(tf.matmul(hidden_output_1, weights_hidden_2), bias_hidden_2)
hidden_output_2 = tf.nn.relu(hidden_input_2)
predictions = tf.add(tf.matmul(hidden_input_2, weights_output), bias_output)

"""
Now we want to define our loss function, in this case will be cross_entropy
"""
softmax_calc = tf.nn.softmax_cross_entropy_with_logits(labels=y_holder, logits=predictions)
cross_entropy = tf.reduce_mean(softmax_calc)  # Cross entropy is actually the cost function

"""
Define which optimizer we want to do, this line will take care of all 
the back propagation and derivatives calculations
"""                              
train_step = tf.train.GradientDescentOptimizer(0.025).minimize(cross_entropy)

# Create session and initialize the variables
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# Run the optimizer
for batch_no in range(10001):
  batch_x, batch_y = mnist.train.next_batch(256)
  
  """
  Here is how this works, essentially we are going to execute our 
  'train_step' value, that will invoke our optimizer, which will then invoke 
  our loss fucntion (cross_entropy) and so on... we need to pass a feed_dict 
  with the VALUES that are needed, ultimately those are X_holder and 
  y_holder as the weights and bias are variables so we cannot really pass them
  """
  train_step.run(feed_dict={X_holder: batch_x, 
                            y_holder: batch_y})
  if batch_no % 250 == 0:
      # Evaluate our accuracy so far
      print("Batch ", batch_no)
      correct_prediction = tf.equal(tf.argmax(y_train,1), tf.argmax(predictions,1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      print(accuracy.eval(feed_dict={X_holder: X_train, y_holder: y_train}))


print("Final accuracy on test set")
correct_prediction = tf.equal(tf.argmax(y_test,1), tf.argmax(predictions,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={X_holder: X_test, y_holder: y_test}))
  

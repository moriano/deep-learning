import tensorflow as tf

# Remove the previous weights and bias
tf.reset_default_graph()

save_file = 'model.ckpt'

# Two Tensor Variables: weights and bias
# Set the name manually so later we can restore the session
weights = tf.Variable(tf.truncated_normal([2, 3]), name='weights_0') 
bias = tf.Variable(tf.truncated_normal([3]), name="bias_0")

saver = tf.train.Saver()

# Print the name of Weights and Bias
print('Save Weights: {}'.format(weights.name))
print('Save Bias: {}'.format(bias.name))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.save(sess, save_file)

# Remove the previous weights and bias
tf.reset_default_graph()

# Two Variables: weights and bias
# Specify the name so we can restore the session correctly
bias = tf.Variable(tf.truncated_normal([3]), name="bias_0")
weights = tf.Variable(tf.truncated_normal([2, 3]), name="weights_0")

saver = tf.train.Saver()

# Print the name of Weights and Bias
print('Load Weights: {}'.format(weights.name))
print('Load Bias: {}'.format(bias.name))

with tf.Session() as sess:
    # Load the weights and bias - ERROR
    saver.restore(sess, save_file)
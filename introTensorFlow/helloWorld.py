import tensorflow as tf
from sklearn.linear_model import LinearRegression

a = LinearRegression()

output = None
x = tf.placeholder(tf.int32)
y = tf.placeholder(tf.int32)

z = tf.add(x, y)
v = tf.Variable(5)
tf.truncated_normal()
n_features = 5
n_labels = 5
weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    # TODO: Feed the x tensor 123
    output = sess.run(z, feed_dict={x: 1, y:2})
    print(weights[0][0])
print(output)
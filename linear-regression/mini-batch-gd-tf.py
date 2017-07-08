'''
    linear regression mini batch gradient descent
    on TensorFlow

    TODO: complete logging
    Author: SayeedM
    Date: 07-07-2014
'''

import numpy as np
import tensorflow as tf
from datetime import datetime
import numpy.random as rnd


def fetch_batch(epoch, batch_index, batch_size):
    rnd.seed(epoch * n_batches + batch_index)
    indices = rnd.randint(m, size = batch_size)
    X_batch = X_b[indices]
    y_batch = y_in[indices]
    return X_batch, y_batch


# setup some logging first so that tensorboard can visualize the data
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_log_dir = "tf_logs"
log_dir = "{}/run-{}".format(root_log_dir, now)

# again lets say we train a naive system to calculate y = 4x - 1
# this time using batch gradient descent
# we will generate a bunch of randoms as training data

X = np.random.rand(1000, 1)
y_in = -1 + 4 * X + np.random.randn(1000, 1) # giving some random noise

# adding a bias (1)
X_b = np.c_[np.ones((1000, 1)), X]

# hyper parameters
eta = 0.1
n_iterations = 1000
m = 1000
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

X = tf.placeholder(shape = (None, 2), dtype = tf.float32, name = "X")
y = tf.placeholder(shape = (None, 1), dtype = tf.float32, name = "y")

# initial theta (random)
theta = tf.Variable( tf.random_uniform([2, 1], -1, 1, seed = 42), name = "theta" )

# define prediction function y = X.thetaT
y_pred = tf.matmul(X, theta, name = "prediction")

with tf.name_scope("loss"):
    # define error function (error = y_predict - y_ideal)
    error = y_pred - y
    # mse function
    mse = tf.reduce_mean(tf.square(error), name = "mse")

optimizer = tf.train.GradientDescentOptimizer(learning_rate = eta)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

# more logging
mse_summary = tf.summary.scalar("MSE", mse)
summary_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())


with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_iterations):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                summary_writer.add_summary(summary_str, batch_index)

            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()

summary_writer.flush()
summary_writer.close()

print("Best theta:")
print(best_theta)

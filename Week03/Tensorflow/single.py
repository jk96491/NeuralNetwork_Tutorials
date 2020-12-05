import tensorflow as tf
import numpy as np

learning_rate = 0.01
training_steps = 10000
display_step = 50


X = [[1], [2], [3]]
Y = [[5], [8], [11]]


W = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bias")

optimizer = tf.optimizers.SGD(learning_rate)


def linear_regression(x):
    return W * x + b


for step in range(1, training_steps + 1):
    # Run the optimization to update W and b values.
    with tf.GradientTape() as tape:
        pred = linear_regression(X)
        loss = tf.reduce_mean(tf.square(pred - Y))

    # Compute gradients.
    gradients = tape.gradient(loss, [W, b])

    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, [W, b]))

    if step % display_step == 0:
        pred = linear_regression(X)
        loss = tf.reduce_mean(tf.square(pred - Y))
        print("step: %i, loss: %f, W: %f, b: %f" % (step, loss, W.numpy(), b.numpy()))


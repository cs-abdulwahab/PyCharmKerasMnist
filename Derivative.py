import tensorflow as tf
import numpy as np
import math


def f(x, y):
    return 4 * x + y * y


xs = np.array([1, 2, 3, 4, 5, 6], dtype="float")

X1 = tf.Variable(5.0, dtype="float")
X2 = tf.Variable(10.0, dtype="float")

# h = 0.000001
# print((f(X1 + h, X2) - f(X1, X2)) / h)

with tf.GradientTape() as tape:
    Y = f(X1, X2)

print(tape.gradient(Y, [X1, X2]))

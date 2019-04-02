import tensorflow as tf
from numpy.random import RandomState


# v = tf.Variable([1, 2])
# init = tf.global_variables_initializer()

# with tf.Session() as sess:
#     sess.run(init)
#     # Usage passing the session explicitly.
#     print(v.eval(sess))
#     # Usage with the default session.  The 'with' block
#     # above makes 'sess' the default session.
#     print(v.eval())


# w1 = tf.Variable(tf.random_normal([1,2], stddev = 1, seed = 1))
# #w2 = tf.Variable(tf.random_normal([1,3], stddev = 1, seed = 1))
# w2 = tf.Variable(w1.initial_value*0.5)

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print(sess.run(w1),sess.run(w2))
# sess.close()

rdm = RandomState(1)
x = rdm.rand(2,3)
y = rdm.rand(2,4)
print(x)
print(y)
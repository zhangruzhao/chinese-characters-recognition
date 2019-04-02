import tensorflow as tf


g1 = tf.Graph()
with g1.as_default():
    v = tf.get_variable("v", initializer = tf.zeros_initializer,shape = [1])
    #v = tf.Variable(name = "v",initial_value = tf.zeros([1]))
    print(v)

g2 = tf.Graph()
with g2.as_default():
    v = tf.get_variable("v", initializer = tf.ones_initializer,shape = [1])
    #print(v)

with tf.Session(graph = g1) as sess:
    sess.run(tf.global_variables_initializer())
    with tf.variable_scope("",reuse = True):
        print(sess.run(tf.get_variable("v")))
        #print(v.eval(sess))
        #v is a variable can not use "print(sess.run(v)"

with tf.Session(graph = g2) as sess:
    sess.run(tf.global_variables_initializer())
    with tf.variable_scope("",reuse = True):
        print(v.eval())
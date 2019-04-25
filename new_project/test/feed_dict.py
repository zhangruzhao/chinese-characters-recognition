import tensorflow as tf

string = tf.placeholder(tf.string)
with tf.Session() as sess:
    string_1 = sess.run(string,feed_dict = {string : './data/tfrecords/*.tfrecord'})
    print(string_1)
import tensorflow as tf
sess = tf.Session()
t = tf.constant([[[1], [1], [1], [2], [2], [2]],
                 [[3], [3], [3], [4], [4], [4]],
                 [[5], [5], [5], [6], [6], [6]]])
#slice1 = tf.strided_slice(t, [1, 0, 0], [2, 1, 3], [1, 1, 1])  # [[[3, 3, 3]]]
slice2 = tf.strided_slice(t, [0, 0, 0], [3, 6, 1], [3, 2, 1])  # [[[3, 3, 3],
                                                               #   [4, 4, 4]]]
#slice3 = tf.strided_slice(t, [1, -1, 0], [2, -3, 3], [1, -1, 1])
print(sess.run(slice2))
import tensorflow as tf

float_image_batch = tf.image.convert_image_dtype(image_batch, tf.float32)

conv2d_layer_one = tf.contrib.layers.convolution2d(
    float_image_batch,
    num_outputs = 32,
    kernels = (5,5),
    activation_fn = tf.nn.relu,
    weight_init = tf.random.normal,
    stride = (2,2)
    trainable = True)

pool_layer_one = tf.nn.max_pool(
    conv2d_layer_one,
    ksize = [1,2,2,1],
    strides = [1,2,2,1],
    padding = 'SAME')

#conv2d_layer_one.get_shape(),pool_layer_one.get_shape()

conv2d_layer_two = tf.contrib.layers.convolution2d(
    pool_layer_one,
    num_outputs = 64,
    kernels = (5,5),
    activation_fn = tf.nn.relu,
    weight_init = tf.random.normal,
    stride = (1,1),
    trainable = True
)

pool_layer_two = tf.nn.max_pool(
    conv2d_layer_two,
    ksize = [1,2,2,1],
    strides = [1,2,2,1],
    padding = 'SAME'
)

flattened_layer_two = tf.reshape(
    pool_layer_two,
    [batch_size, -1]#batch_size = 3,-1为输入的其他维调整为一个巨大的秩1的张量
)

#全连接层
 hidden_layer_three = tf.contrib.layers.fully_connected(
     flattened_layer_two,
     num_outputs = 512,
     activation_fn = tf.nn.relu,
     weight_init = lambda i ,dtype: tf.truncated_normal([38912,512],stddev = 0.1)
 )
 
 hidden_layer_three = tf.nn.dropout(hidden_layer_three,0.1)

 final_fully_connected = tf.contrib.layers.fully_connected(
     hidden_layer_three,
     num_outputs = 120,
     weight_init = lambda i, dtype: tf.truncated_normal([512.120], stddev = 0.1),
 )
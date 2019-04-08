import tensorflow as tf

batch_size = 3
def maxout(conv, num_units):
    return tf.contrib.layers.maxout(conv = conv, num_units = num_units)

def conv2d_layer_maxout(inputs, num_output_channels, kernel_size, stride = (1,1), padding = 'SAME',num_units):
    conv2d_layer = tf.contrib.layers.conv2d(
        inputs = inputs,
        num_outputs=num_output_channels,
        kernel_size=kernel_size,
        #weights_initializer=tf.random_normal,
        stride = stride,
        padding=padding,
        trainable=True
    )
    maxout = maxout(couv2d_layer,num_units);
    return maxout

def cnnModel(inputs):
    conv2d_maxout_one = conv2d_layer_maxout(
        inputs = inputs,
        num_output_channels=96,
        kernel_size=(9,9),
        num_units=48
    )
    
    conv2d_maxout_two = conv2d_layer_maxout(
        inputs=conv2d_maxout_one,
        num_output_channels=128,
        kernel_size=(9,9),
        num_units=64
    )

    conv2d_maxout_three = conv2d_layer_maxout(
        inputs=conv2d_maxout_two,
        num_output_channels=256,
        kernel_size=(9,9),
        num_units=128
    )

    conv2d_maxout_four = conv2d_layer_maxout(
        inputs=conv2d_maxout_three,
        num_output_channels=512,
        kernel_size=(8,8),
        num_units=128
    )

    conv2d_maxout_five = conv2d_layer_maxout(
        inputs=conv2d_maxout_four,
        num_output_channels=144
        kernel_size=(1,1),
        num_units=36
    )

    return conv2d_maxout_five
    
def rnn(inputs,seq_lenth):

def slidingImage(image_batch,wid):
    image_list = tf.split(0,image_batch,batch_size)
    image_slice_batch_list = []
    image_slice_expend_list = []
    for i in range(50):
        for batch_ind in range(batch_size):
            image_slice = tf.slice(image_list[batch_ind],[i*100,i*40,0],[100,40,1])
            image_slice_expend = tf.expand_dims(image_slice,0)
            image_slice_expend_list.append(image_slice_expend)
        image_slice_batch = tf.concat(0,image_slice_expend_list)
        image_slice_batch_list.append(image_slice_batch)
    return image_slice_batch_list

def train():
    image_slice_batch_list = slidingImage(image_batch,wid)
    cnn_slice_output_list = []
    for i in range(wid):
        cnn_slice_output = cnnModel(image_slice_batch_list[i])
        cnn_slice_output_list.append(cnn_slice_output)
    cnn_output = tf.concat(2,cnn_slice_output_list)
    


            
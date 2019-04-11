import tensorflow as tf


def Maxout(conv, num_units):
    return tf.contrib.layers.maxout(inputs = conv, num_units = num_units)

def conv2d_layer_maxout(inputs, num_output_channels, kernel_size, num_units, stride = (1,1), padding = 'VALID'):
    conv2d_layer = tf.contrib.layers.conv2d(
        inputs = inputs,
        num_outputs=num_output_channels,
        kernel_size=kernel_size,
        stride = stride,
        padding=padding,
        trainable=True
    )
    maxout = Maxout(conv2d_layer,num_units);
    return maxout

def cnnModel(inputs):
    conv2d_maxout_zero = conv2d_layer_maxout(
        inputs = inputs,
        num_output_channels=48,
        kernel_size=(69,9),
        num_units = 24
    )
    conv2d_maxout_one = conv2d_layer_maxout(
        inputs = conv2d_maxout_zero,
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
        num_output_channels=144,
        kernel_size=(1,1),
        num_units=36
    )

    return conv2d_maxout_five
    
def rnn(inputs,seq_lenth):
    pass

def slidingImage(image_batch,batch_size):
    image_list = tf.split(axis=0,value=image_batch,num_or_size_splits=batch_size)
    image_slice_batch_list = []
    for i in range(50):
        image_slice_expend_list = []
        for batch_ind in range(batch_size):
            image_slice = tf.slice(image_list[batch_ind][0],[0,i*40,0],[100,40,1])
            image_slice_expend = tf.expand_dims(image_slice,0)#4d
            image_slice_expend_list.append(image_slice_expend)
        image_slice_batch = tf.concat(axis = 0,values=image_slice_expend_list)
        image_slice_batch_list.append(image_slice_batch)
    return image_slice_batch_list

def train(image_batch,batch_size):
    image_slice_batch_list = slidingImage(image_batch,batch_size)
    cnn_slice_output_list = []
    for i in range(50):
        cnn_slice_output = cnnModel(image_slice_batch_list[i])
        cnn_slice_output_list.append(cnn_slice_output)
    print(cnn_slice_output_list)
    cnn_output = tf.concat(axis=2,values=cnn_slice_output_list)
    print(cnn_output)
    


            
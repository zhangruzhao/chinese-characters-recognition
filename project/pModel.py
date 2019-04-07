import tensorflow as tf

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

def pModel(inputs):
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

def slidingImg(image,wid):
    length = round(wid/40)
    resized_image = tf.image.resize_image_with_crop_or_pad(image,target_height=100,target_width=length*40)

    for i in 
        
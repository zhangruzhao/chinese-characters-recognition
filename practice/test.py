import tensorflow as tf

image_input = tf.constant([
    [#第一个输入input_high*input_width*input_channels 3*3*3
        [[0.,0.,0.,],[255.,255.,255.],[254.,0.,0.]],#第一行
        [[0.,191.,0.],[3.,108.,233.],[0.,191.,0]],#第二行
        [[254.,0.,0.],[255.,255.,255.],[0.,0.,0]]#第三行
    ]
])

conv2d = tf.contrib.layers.convolution2d(
    image_input, num_outputs = 4,#输出通道数
    kernel_size = (1,1),#这里只有卷积核的高度和宽度
    activation_fn = tf.nn.relu,
    stride = (2,2),#对width 和 high 的跨度
    trainable = True
)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(conv2d))
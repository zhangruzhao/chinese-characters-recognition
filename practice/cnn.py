import tensorflow as tf

inputs = tf.constant([ #inputs[batch_size,in_height,in_width,in_channels]
    [
        [[1.,1.,1.]]#,[2.,2.,2.],[3.,3.,3.]],
        #[[1.,2.,3.],[2.,3.,4.],[3.,4.,5.]]
    ]
])
#kernel的形状中的in_channels 要和inputs形状中的in——channels相等。
kernel = tf.constant([ #kernel [kernel_height,k_width,in_channels,out_channels]
    [
        [[1.,1.,3.,4.],[1.,2.,3.,4.],[1.,2.,3.,4.]]
    ]
])
couv2d = tf.nn.conv2d(inputs,kernel,strides = [1,1,1,1],padding = 'SAME')
maxout = tf.contrib.layers.maxout(couv2d,2)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(couv2d))#[[[[3. 5. 9.]]]]
    #print(sess.run(maxout))
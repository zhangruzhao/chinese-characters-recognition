import tensorflow as tf

class CRNNModel(object):
    def __init__(self):
        return
    def Maxout(self,conv, num_units):
        return tf.contrib.layers.maxout(inputs = conv, num_units = num_units)

    def conv2d_layer_maxout(self,inputs, num_output_channels, kernel_size, num_units, stride = (1,1), padding = 'VALID'):
        conv2d_layer = tf.contrib.layers.conv2d(
            inputs = inputs,
            num_outputs=num_output_channels,
            kernel_size=kernel_size,
            stride = stride,
            padding=padding
        )
        maxout = self.Maxout(conv2d_layer,num_units);
        return maxout

    def cnnModel(self,inputs):
        with tf.variable_scope('cnnModel',reuse=tf.AUTO_REUSE):
            conv2d_maxout_zero = self.conv2d_layer_maxout(
                inputs = inputs,
                num_output_channels=48,
                kernel_size=(69,9),
                num_units = 24
            )
            conv2d_maxout_one = self.conv2d_layer_maxout(
                inputs = conv2d_maxout_zero,
                num_output_channels=96,
                kernel_size=(9,9),
                num_units=48
            )
            conv2d_maxout_two = self.conv2d_layer_maxout(
                inputs=conv2d_maxout_one,
                num_output_channels=128,
                kernel_size=(9,9),
                num_units=64
            )
            conv2d_maxout_three = self.conv2d_layer_maxout(
                inputs=conv2d_maxout_two,
                num_output_channels=256,
                kernel_size=(9,9),
                num_units=128
            )
            conv2d_maxout_four = self.conv2d_layer_maxout(
                inputs=conv2d_maxout_three,
                num_output_channels=512,
                kernel_size=(8,8),
                num_units=128
            )

            # conv2d_maxout_five = conv2d_layer_maxout(
            #     inputs=conv2d_maxout_four,
            #     num_output_channels=144,
            #     kernel_size=(1,1),
            #     num_units=36
            # )

        return conv2d_maxout_four
        
    def rnn(self,inputs,seq_length,batch_size,num_classes):
        with tf.variable_scope('LSTM_Layers'):
            inputs_reshaped = tf.reshape(inputs,[batch_size,50,128],name='input_reshaped')
            lstm_units = 64 #lstm_units*batch_size == lstm_units+128
            layers_num = 3
            lstm = tf.contrib.rnn.BasicLSTMCell(lstm_units,forget_bias=1.0)
            fw_cell_list = [lstm for i in range(layers_num)]
            bw_cell_list = [lstm for i in range(layers_num)]
            stack_lstm_layers,_,_ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                fw_cell_list,bw_cell_list,inputs_reshaped,sequence_length=seq_length,dtype=tf.float32
            )

            print("stack_lstm_layers:",stack_lstm_layers)
            hidden_num = int(stack_lstm_layers.get_shape()[2])
            print("hidden_num:",hidden_num)
            rnn_reshaped = tf.reshape(stack_lstm_layers,[-1,hidden_num])
            weight = tf.Variable(tf.truncated_normal([hidden_num,num_classes],stddev=0.01),name='weights')
            logits = tf.matmul(rnn_reshaped, weight,name='logits')

            logits = tf.reshape(logits,[batch_size, -1, num_classes],name='logits_reshaped')
            raw_pred = tf.argmax(tf.nn.softmax(logits),axis=2,name='raw_pred')

            rnn_out = tf.transpose(logits,(1,0,2))#[max_length,batch_size,num_classes]
        return rnn_out

    def slidingImage(self,image_batch,batch_size):
        with tf.variable_scope('slideImage'):
            image_list = tf.split(axis=0,value=image_batch,num_or_size_splits=batch_size)
            image_slice_batch_list = []
            for i in range(50):
                image_slice_expend_list = []
                for batch_ind in range(batch_size):
                    image_slice = tf.slice(image_list[batch_ind][0],[0,i*40,0],[100,40,1])
                    image_slice_expend = tf.expand_dims(image_slice,0)#4d
                    image_slice_expend_list.append(image_slice_expend)
                image_slice_batch = tf.concat(axis = 0,values=image_slice_expend_list,name='concat_on_batch')
                image_slice_batch_list.append(image_slice_batch)
        return image_slice_batch_list

    def train(self,image_batch,batch_size,sequence_lengths,num_classes):
        with tf.variable_scope('build_net'):
            image_slice_batch_list = self.slidingImage(image_batch,batch_size)
            with tf.variable_scope('cnn_output'):
                cnn_slice_output_list = []
                for i in range(50):
                    cnn_slice_output = self.cnnModel(image_slice_batch_list[i])
                    cnn_slice_output_list.append(cnn_slice_output)
                #print(cnn_slice_output_list)
                cnn_output = tf.concat(axis=2,values=cnn_slice_output_list,name='cnn_output')
                print("cnn_output:",cnn_output)

            ###
            rnn_output = self.rnn(cnn_output,seq_length=sequence_lengths,batch_size=batch_size,num_classes=num_classes)
            print("rnn_output:",rnn_output)
        return rnn_output
    
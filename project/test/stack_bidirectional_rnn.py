            # stack_lstm = tf.contrib.rnn.MultiRNNCell([lstm]*layers_num)
            # def _bidirecional_rnn(data,length):
            #    length_64 = tf.cast(length,tf.int64)
            #     forward, state_f = tf.nn.dynamic_rnn(
            #         cell=stack_lstm,
            #         inputs=data,
            #         sequence_length=length,
            #         dtype=tf.float32,
            #         scope='rnn_forward'
            #     )
            #     backward, state_b = tf.nn.dynamic_rnn(
            #         cell=stack_lstm,
            #         inputs=tf.reverse_sequence(data,length_64,seq_dim=1),
            #         sequence_length=length,
            #         dtype=tf.float32,
            #         scope='rnn__backward'
            #     )
            #     backward = tf.reverse_sequence(backward, length_64, seq_dim=1)
            #     output = tf.concat(axis=2,values=[forward,backward])
            #     return output
            
            # bid_output = _bidirecional_rnn(inputs_reshaped,seq_length)
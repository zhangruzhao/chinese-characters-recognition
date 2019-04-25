import tensorflow as tf
import numpy as np

LEARNING_RATE = 0.1
DECAY_STEP = 1000
DECAY_RATE = 0.8

def sparse_matrix_from(labels):
    with tf.variable_scope('sparse_matrix_from'):
        indices = []
        values = []
        for batch_ind, label_line in enumerate(labels):
            indices.extend(zip([batch_ind]*len(label_line),range(len(label_line))))
            values.extend(label_line)
        
        indices = np.asarray(indices,dtype = np.int32)
        values = np.asarray(values,dtype = np.int32)
        shape = np.asarray([len(labels),indices.max(0)[1]+1],dtype = np.int32)
        # sparse_matrix = tf.SparseTensor(indices,values,shape)
        # return sparse_matrix
    return indices, values, shape

def ctc_loss_train(sparse_labels,inputs,seq_length):
    with tf.variable_scope('ctc_loss_train'):

        ctc_input = inputs
        ctc_seq_length = seq_length

        global_step = tf.train.create_global_step()
        learning_rate = tf.train.exponential_decay(
            learning_rate=LEARNING_RATE,
            global_step=global_step,
            decay_steps=DECAY_STEP,
            decay_rate=DECAY_RATE,
            staircase=True
        )

        loss = tf.nn.ctc_loss(
            labels=sparse_labels,
            inputs=ctc_input,
            sequence_length=ctc_seq_length,
            ignore_longer_outputs_than_inputs=True
        )

        ctc_loss = tf.reduce_mean(loss)
        ctc_decoded, ctc_log_prob = tf.nn.ctc_beam_search_decoder(ctc_input,ctc_seq_length,merge_repeated=False)
        seq_distance = tf.reduce_mean(tf.edit_distance(tf.cast(ctc_decoded[0],tf.int32),sparse_labels))
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(
                loss=ctc_loss,
                global_step=global_step
            )

    return optimizer,ctc_loss,learning_rate,seq_distance,ctc_decoded
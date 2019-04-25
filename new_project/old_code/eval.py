import tensorflow as tf
import pModel
import load_image as loadImg
import glob
import ctc_train
import numpy as np
import os
import time
import Model_test

# def get_next_batch(batch_size):
#     with tf.variable_scope('get_next_batch'):
#         image_list = []
#         label_batch = []
#         word_num_batch = []
#         hei_batch = []
#         wid_batch = []
#         for i in range(batch_size):
#             image, label, word_num, hei, wid = iterator.get_next()
#             #image_batch
#             image_float = tf.image.convert_image_dtype(image, dtype=tf.float32,name='image_float')
#             image_expend = tf.expand_dims(image_float,0,name='image_4d')
#             image_list.append(image_expend)

#             label_batch.append(label)
#             word_num_batch.append(word_num)
#             wid_batch.append(wid)
#         image_batch = tf.concat(axis=0,values=image_list,name='image_batch')
#     return image_batch, label_batch, word_num_batch, wid_batch

def convert_labels_to_int(label_batch, word_num_batch, batch_size):
    image_labels = []
    for batch_ind in range(batch_size):
        label_line_int = []
        label = label_batch[batch_ind]
        word_num = word_num_batch[batch_ind]
        for i in range(word_num):
            label_word = label[2*i:2*(i+1)]

            # print("label_word:",label_word,label_word.decode('gb2312'))
            label_int = int.from_bytes(label_word,byteorder='little')
            label_line_int.append(label_int)
        image_labels.append(label_line_int)
    #sparse_labels = ctc_train.sparse_matrix_from(image_labels)
    return image_labels

def get_seq_len(wid_batch,batch_size):
    sequence_lengths = []
    for i in range(batch_size):
        wid = wid_batch[i]
        rou_len = round(wid/40)
        seq_length = 25+int(round(rou_len/2)) #rou_len+(50-rou_len)/2  
        sequence_lengths.append(seq_length)
    return sequence_lengths

def sparse_to_list(sparse_matrix):
    indices = sparse_matrix.indices
    values = sparse_matrix.values
    dense_shape = sparse_matrix.dense_shape

    dense_matrix = (num_classes-1)*np.ones(dense_shape,dtype=np.int32)
    for ind, indice in enumerate(indices):
        dense_matrix[indice[0],indice[1]] = values[ind]
    string_list = []
    for row in dense_matrix:
        string = []
        for val in row:
            string.append(val)
        string_list.append(string)
    return string_list

def get_accuracy(preds, lab):
    preds = sparse_to_list(preds[0])
    accuracy = []
    for index, line_label in enumerate(lab):
        pred = preds[index]
        total_count = len(line_label)
        correct_count = 0
        try:
            for i, label in enumerate(line_label):
                if label == pred[i]:
                    correct_count += 1
        except IndexError:
            continue
        finally:
            try:
                accuracy.append(correct_count/total_count)
            except ZeroDivisionError:
                if len(pred) == 0:
                    accuracy.append(1)
                else:
                    accuracy.append(0)
    accuracy = np.mean(np.array(accuracy).astype(np.float32),axis=0)
    return accuracy

num_classes = 3042 #including blank
batch_size = 20
model_dir = './model_1/'

filenames = tf.placeholder(tf.string,shape=[None],name='filenames')
next_batch = tf.placeholder(dtype=tf.float32,shape=[batch_size,32,None,1],name='next_batch')
sequence_lengths = tf.placeholder(dtype=tf.int32,shape=None,name='sequence_lengths')
labels = tf.sparse_placeholder(dtype=tf.int32,shape=None,name='labels')

with tf.variable_scope('record_dataset'):
    record_dataset = tf.data.TFRecordDataset(filenames)
    record_dataset = record_dataset.map(loadImg.parse_serialized)
    batch_dataset = record_dataset.shuffle(buffer_size = 10).batch(20)

    iterator = batch_dataset.make_initializable_iterator()

# image_batch, label_batch, word_num_batch, wid_batch = get_next_batch(batch_size)
# model = pModel.CRNNModel()
# model_output = model.train(next_batch,batch_size, sequence_lengths, num_classes)

image_batch, label_batch, word_num_batch, hei_batch, wid_batch = iterator.get_next()
model = Model_test.CRNNCTCNetwork('train',256,2,num_classes)
with tf.variable_scope('crnn',reuse=False):
    model_output = model.build_network(next_batch,sequence_lengths)

ctc_decoded, ct_log_prob = tf.nn.ctc_beam_search_decoder(model_output, sequence_lengths, merge_repeated=False)
# optimizer, ctc_loss, learning_rate, sequence_distance, ctc_decoded = ctc_train.ctc_loss_train(
#     labels,model_output,sequence_lengths)

# set checkpoint saver
saver = tf.train.Saver()
save_path = tf.train.latest_checkpoint(model_dir)

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
with tf.Session(config=sess_config) as sess:
    saver.restore(sess=sess, save_path=save_path)
    eval_filenames = glob.glob(
    "/home/zhang/桌面/HIT-MW database/02 HIT-MW GT (binary)/tfrecord/*.tfrecords")

    eval_filenames = eval_filenames[0:20]
    sess.run(iterator.initializer,feed_dict = {filenames:eval_filenames})

    step = 0
    try:
        while 1:
            im_bat, lab_bat, word_num_bat, wid_bat = sess.run([image_batch, label_batch, word_num_batch, wid_batch])
            lab_int = convert_labels_to_int(lab_bat, word_num_bat, batch_size)
            lab = ctc_train.sparse_matrix_from(lab_int)
            seq_len = get_seq_len(wid_bat, batch_size)

            preds,prob = sess.run([ctc_decoded,ct_log_prob],
                feed_dict={next_batch:im_bat, sequence_lengths:seq_len, labels:lab})
            preds = sparse_to_list(preds[0])

            for index, img in enumerate(im_bat):
                print("step:", step,"labels:",lab_int[index],"preds:",preds[index])
            
            step = step + 1
    except tf.errors.OutOfRangeError:
        print("end!")
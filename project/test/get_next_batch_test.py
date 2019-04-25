# def get_next_batch(iterator,batch_size):
#     with tf.variable_scope('get_nextbatch'):
#         def processing_labels(label,word_num):
#             with tf.variable_scope('processing_labels'):
#                 label_line_int = []
#                 for i in range(word_num):
#                     label_word = label[2*i:2*(i+1)]
#                     label_int = int.from_bytes(label_word,byteorder='little')
#                     label_line_int.append(label_int)
#             return label_line_int
        
#         image_expend_list = []
#         sequence_lengths = []
#         labels_int = []
#         for i in range(batch_size):
#             image, label, word_num, hei, wid = iterator.get_next()
#             image_float = tf.image.convert_image_dtype(image, dtype=tf.float32)

#             image_expend = tf.expand_dims(image_float,0)
#             image_expend_list.append(image_expend)

#             label, word_num, hei, wid = sess.run([label, word_num, hei, wid])
#             rou_len = round(wid/40)
#             seq_length = 25+int(round(rou_len/2)) #rou_len+(50-rou_len)/2  
#             sequence_lengths.append(seq_length)
            
#             labels_int.append(processing_labels(label, word_num))
#             # [[line_0_label_0,line_0_label_1], line_0
#             #  [line_1_label_0,line_1_label_1], line_1
#             #  [line_2_label_0,line_2_label_1]] line_2
#         next_batch = tf.concat(axis=0,values=image_expend_list)
#         print("next_batch:",next_batch)

#         sparse_labels = ctc_train.sparse_matrix_from(labels_int)
#         next_bat = sess.run(next_batch)
#     return next_bat, sequence_lengths, sparse_labels
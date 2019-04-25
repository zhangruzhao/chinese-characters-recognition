#image, label, word_num, hei, wid = iterator.get_next()

#record_dataset.shuffle(buffer_size = 10).batch(batch_size = 3)

# filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(
#     "/home/zhang/桌面/HIT-MW database/02 HIT-MW GT (binary)/tfrecord/b04031701-0.tfrecords"
# ))
# reader = tf.TFRecordReader()
# _, serialized = reader.read(filename_queue)
# image, labels, word_num = parse_serialized(serialized)
# with tf.Session() as sess:
#     init = (tf.local_variables_initializer(),tf.global_variables_initializer())
#     sess.run(init)
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord= coord)
#     lab, wo_num = sess.run([labels,word_num])
#     print(lab,wo_num)
#     coord.request_stop()
#     coord.join(threads)

# min_after_dequeue = 10
# batch_size = 3
# capacity = min_after_dequeue + 3*batch_size
# image_batch, label_batch = tf.train.shuffle_batch([image. labels],batch_size=batch_size,capacity=capacity,min_after_dequeue=min_after_dequeue)
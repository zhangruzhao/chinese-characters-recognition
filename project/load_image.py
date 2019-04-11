import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import numpy as np

def parse_serialized(example):
    features = tf.parse_single_example(
        example,
        features = {
            'word_num':tf.FixedLenFeature([],tf.int64),
            'label_len':tf.FixedLenFeature([],tf.int64),
            'hei':tf.FixedLenFeature([],tf.int64),
            'wid':tf.FixedLenFeature([],tf.int64),
            'label': tf.FixedLenFeature([],tf.string),
            'image': tf.FixedLenFeature([],tf.string),
        }
    )
    image = tf.decode_raw(features['image'], tf.uint8)
    hei = tf.cast(features['hei'],tf.int32)
    wid = tf.cast(features['wid'],tf.int32)
    word_num = tf.cast(features['word_num'],tf.int32)

    resized_image = resize_image(image,hei,wid)

    label = tf.cast(features['label'], tf.string)
    return  resized_image, label, word_num, hei, wid

def resize_image(image,hei,wid):
    reshaped_image = tf.reshape(image, [hei,wid,1])
    
    round_wid = tf.round(wid/40)
    resized_image = tf.image.resize_images(reshaped_image,[100,round_wid*40])
    final_image = tf.image.resize_image_with_crop_or_pad(resized_image,100,2000)
    return final_image

def get_next_batch(iterator,batch_size):
    # def priImg(data):
    #     plt.figure()
    #     plt.imshow(data)
    #     plt.show()

    image_expend_list = []
    for i in range(batch_size):
        image, label, word_num, hei, wid = iterator.get_next()
        #image_data = tf.reshape(image,[100,2000])
        #image_data = sess.run(image_data)
        #print(image_data.shape)
        #priImg(image_data)
        image_expend = tf.expand_dims(image,0)
        image_expend_list.append(image_expend)
    next_batch = tf.concat(axis=0,values=image_expend_list)
    #print(next_batch)
    return next_batch

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


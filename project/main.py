import tensorflow as tf
import pModel
import load_image as loadImg
import glob

batch_size = 3
filenames = tf.placeholder(tf.string,shape=[None])
record_dataset = tf.data.TFRecordDataset(filenames)
record_dataset = record_dataset.map(loadImg.parse_serialized)
batch_dataset = record_dataset.shuffle(buffer_size = 10)

iterator = batch_dataset.make_initializable_iterator()
init = (tf.local_variables_initializer(),tf.global_variables_initializer())
with tf.Session() as sess:
    train_filenames = glob.glob(
    "/home/zhang/桌面/HIT-MW database/02 HIT-MW GT (binary)/tfrecord/*.tfrecords")
    train_filenames = train_filenames[0:6]
    sess.run(init)
    sess.run(iterator.initializer,feed_dict = {filenames:train_filenames})
    i = 0
    try:
        while 1:
            next_batch = sess.run(loadImg.get_next_batch(iterator,batch_size))
            #print(next_batch)
            pModel.train(next_batch,batch_size)
    except tf.errors.OutOfRangeError:
        print("end!")
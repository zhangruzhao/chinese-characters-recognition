import tensorflow as tf

#预处理为TFRecord文件
image_label = b'\x01'#独热编码

image_loaded = sess.run(image)#加载到内存中
image_bytes = image_loaded.tobytes()#转换为字节数组

image_height, image_width, image_channels = image_loaded.shape

#新建一个writer 
writer = tf.python_io.TFRecordWriter("./output/training-image.tfrecord")

#将字节添加到Example文件中
example = tf.train.Example(features = tf.train.Features(feature = {
    'label': tf.train.Feature(bytes_list = tf.train.BytesList(value = [image_label])),
    'image': tf.train.Feature(bytes_list = tf.train_BytesList(value = [image_bytes]))
}))
#将Example 文件存入磁盘前，将其序列化为二进制字符串，再将用writer 将样本写入 tfrecord 文件中
writer.write(example.SerializeToString())
writer.close()

#读取tfrecord文件
tf_record_filenames_queue = tf.data.DataSet.from_tensor_slices(tf.train.match_filenames_once(
    "./output/training-image.tfrecord"
))

tf_record_reader = tf.TFRecordReader()
_,tf_record_files = tf_record_reader.read(tf_record_filenames_queue)

#解析example protobuf文件
tf_record_features = tf.parse_single_example(
    tf_record_files,
    features = {
        'label':tf.FixedLenFeature([],tf.String),
        'image': tf.FixedLenFeature([],tf.String)
    }
)
#解码tfrecord 文件
tf_record_image = tf.decode_raw(tf_record_features['image'],tf.uint8)#将原始字节解码成unit8类型

#调整图像的尺寸，使其与所保存的图像类似，但这不是必须的
tf_record_image = tf.reshape(tf_record_image,[image_height,image_width,image_channels])

tf_record_label = tf.cast(tf_record_features['label'],tf.string)#将字节转换为字符串

'''
image_filenames = ""

#filenames_queue = tf.train.string_input_producer(tf.train.match_filenames_once(image_filenames))(deprecated)
filenames_queue = tf.data.DataSet.from_tensor_slice(tf.train.match_filenames_once(image_filenames))

reader = tf.WholeFileReader()
#The output of Read will be a filename (key) and the contents of that file (value)
_, image_file = reader.read(filenames_queue)
image = tf.image.decode_jpeg(image_file)

sess = tf.Session()
sess.run(image)
sess.close()
'''
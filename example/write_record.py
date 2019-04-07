import tensorflow as tf

def write_record_file(dataset, record_location):
    writer = None
    current_index = 0

    for breed, images_filenames in dataset.items():
        for images_filename in images_filenames:
            if current_index%100 == 0:
                if writer:
                    writer.close()
                record_file = "{record_location}-{current_index}.tfrecords".format(
                    record_location = record_location,
                    current_index = current_index
                )
                writer = tf.python_io.TFRecordWriter(record_file)

            image_file = tf.read_file(images_filename)
            # reader = tf.WholeFileReader()
            # _,image_file = reader.read(images_filename)
            try:
                image = tf.image.decode_jpeg(image_file)
            except:
                print(image_file)
                continue
            
            grayscale_image = tf.image.rgb_to_grayscale(image)
            resized_image = tf.image.resize_images(grayscale_image,[250,151])

            #使用tf.cast是因为尺寸更改后图像的数据类型是浮点型，但RGB值尚未转换到[0,1)区间
            #with tf.Session() as sess:
            image_bytes = sess.run(tf.cast(resized_image,tf.uint8)).tobytes()
            image_label = breed.encode("utf-8")

            example = tf.train.Example(features = tf.train.Features(feature = {
                'label': tf.train.Feature(bytes_list = tf.train.BytesList(value = [image_label])),
                'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [image_bytes]))
            }))

            writer.write(example.SerializeToString())
            writer.close()





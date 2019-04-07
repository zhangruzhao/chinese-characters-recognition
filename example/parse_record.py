import tensorflow as tf

def parse_serialized(example):
    features = tf.parse_single_example(
        example,
        features = {
            'label': tf.FixedLenFeature([],tf.string),
            'image': tf.FixedLenFeature([],tf.string)
        }
    )
    image = tf.decode_raw(features['image'], tf.uint8)
    resized_image = tf.reshape(image, [250, 151, 1])
    label = tf.cast(features['label'], tf.string)
    return resized_image, label

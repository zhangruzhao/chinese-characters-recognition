import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import numpy as np

def parse_serialized(example):
    with tf.variable_scope('parse_serialized'):
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

        #resized_image = resize_image(image,hei,wid)
        resized_image = resize_image_test(image,hei,wid)

        label = tf.cast(features['label'], tf.string)
    return  resized_image, label, word_num, hei, wid

def resize_image(image,hei,wid):
    with tf.variable_scope('resize_image'):
        reshaped_image = tf.reshape(image, [hei,wid,1])
        
        round_wid = tf.round(wid/40)
        resized_image = tf.image.resize_images(reshaped_image,[100,round_wid*40])
        final_image = tf.image.resize_image_with_crop_or_pad(resized_image,100,2000)
    return final_image

def resize_image_test(image,hei,wid):
    with tf.variable_scope('resize_image'):
        reshaped_image = tf.reshape(image, [hei,wid,1])
        resized_image = tf.image.resize_images(reshaped_image,[32,512])
    return resized_image


"""
Write text features and labels into tensorflow records
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import random
import glob

import tensorflow as tf

import cv2
import numpy as np

_IMAGE_HEIGHT = 32
_IMAGE_WIDTH = 400


# tf.app.flags.DEFINE_string(
#     'image_dir', './dataset/images/', 'Dataset root folder with images.')

# tf.app.flags.DEFINE_string(
#     'anno_file', './dataset/anno_file.txt', 'Path of dataset annotation file.')

tf.app.flags.DEFINE_string(
    'data_dir', './tfrecords/', 'Directory where tfrecords are written to.')

# tf.app.flags.DEFINE_float(
#     'validation_split_fraction', 0.1, 'Fraction of training data to use for validation.')

# tf.app.flags.DEFINE_boolean(
#     'shuffle_list', True, 'Whether shuffle data in annotation file list.')

# tf.app.flags.DEFINE_string(
#     'char_map_json_file', './char_map/char_map.json', 'Path to char map json file') 

FLAGS = tf.app.flags.FLAGS

def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _string_to_int(label,char_dict):
    # convert string label to int list by char map

    
    int_list = []
    for c in label:
        try:
            int_list.append(char_dict[c])
        except KeyError:
            print("keyerror:",c)
            int_list.append(char_dict[c[-1]])
    return int_list

def _write_tfrecord(filenames,char_dict):
    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)
    
    # 
    
    tfrecords_path = os.path.join(FLAGS.data_dir, 'train' + '.tfrecord')

    with tf.python_io.TFRecordWriter(tfrecords_path) as writer:
        for filename in filenames:
            image_path = "./bmp/{filename}.bmp".format(filename = filename)
            label_path = "./line_txt/{filename}.txt".format(filename = filename)

            image = cv2.imread(image_path)
            if image is None: 
                continue # skip bad image.
            # image = cv2.resize(image, _IMAGE_SIZE)

            h, w, c = image.shape
            # if c != 1:
            #     print("shape-c:",c)

            height = _IMAGE_HEIGHT
            #width = int(w * height / h)
            width = _IMAGE_WIDTH
            image = cv2.resize(image, (width, height))

            cv2.imshow(filename,image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            is_success, image_buffer = cv2.imencode('.bmp', image)

            if not is_success:
                print("imencoded_failed:",filename)
                continue

            label_file = open(label_path,'r')
            label = label_file.read()
            label_list = label.split(" ")[1:]
            #print("label_list:",label_list)

            seq_len = len(label_list)
            #print(seq_len)

            # # convert string object to bytes in py3
            image_name =  filename.encode('utf-8') 
            
            features = tf.train.Features(feature={
            'labels': _int64_feature(_string_to_int(label_list,char_dict)),
            'images': _bytes_feature(image_buffer.tostring()),
            'seq_len': _int64_feature(seq_len),
            'image_name':_bytes_feature(image_name)
            })
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())

        sys.stdout.write('>> {:s}.tfrecords write finish.'.format(filename))
        sys.stdout.write('\n')
        sys.stdout.flush()

def _convert_dataset():
    # with open(FLAGS.anno_file, 'r') as anno_fp:
    #     anno_lines = anno_fp.readlines()

    file = open(r'./char_set_utf.txt','r')
    char_set = file.read()
    char_list = list(char_set)
    #print(char_list)
    char_dict = dict(zip(char_list,range(len(char_list))))
    #print(char_dict)

    image_filenames = glob.glob(r'./bmp/b0401010*.bmp')
    filenames = map(lambda filenames: filenames.split(r'/')[-1][:-4],image_filenames)

    _write_tfrecord(filenames,char_dict)

    file.close()

def main(unused_argv):
    _convert_dataset()

if __name__ == '__main__':
    tf.app.run()

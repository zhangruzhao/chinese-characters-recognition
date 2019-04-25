import tensorflow as tf
from preProcessing import Image
import numpy as np
import glob

def write_record():
    with tf.variable_scope('write_record'):
        image_filenames = glob.glob("/home/zhang/桌面/HIT-MW database/02 HIT-MW GT (binary)/binary_format/*.dgr")
        #image_filenames = image_filenames[0:2]

        ind = 0
        max_line_length = 0
        for image_filename in image_filenames:
            print("image:",ind)
            ind = ind + 1

            image = Image(image_filename)
            label_len = image.get_codeLen()
            current_ind = image_filename.split("/")[-1][:-4]
            line_num = image.get_lineNum()
            dgr_header = image.get_DGR_HEADER()

            for line_ind in range(line_num):
                record_filename = "{record_location}{current_index}-{line_index}.tfrecords".format(
                    record_location = r'/home/zhang/桌面/HIT-MW database/02 HIT-MW GT (binary)/tfrecord/',
                    current_index = current_ind,
                    line_index = line_ind)
                writer = tf.python_io.TFRecordWriter(record_filename)
                line_hei = image.get_lineShape(lineInd=line_ind)['Hei']
                line_wid = image.get_lineShape(lineInd=line_ind)['Wid']
                if line_wid > max_line_length:
                    max_line_length = line_wid
                line_labels,line_image = image.get_lineData(lineInd=line_ind)

                #print(line_image.shape,line_hei,line_wid)
                #line image是二维np array，将它转换为bytes字符串
                line_img_list = line_image.flatten().tolist()
                line_bytes = bytes(line_img_list)
                #print(len(line_bytes))
                
                #line labels是bytes数组list，将它转为bytes字符串 ：先解码unicode，才能转为str，转为str后，str再编码为bytes
                try:
                    line_labels = [x.decode('gb2312') for x in line_labels]
                    word_num = len(line_labels)
                    line_labels_str = ''.join(line_labels)
                    line_labels_str = line_labels_str.encode('gb2312')
                except:
                    print("error_code_type:",dgr_header['CodeType'])
                    continue


                #print(line_labels_str)
                example = tf.train.Example(features = tf.train.Features(
                    feature = {
                        'word_num':tf.train.Feature(int64_list = tf.train.Int64List(value = [word_num])),
                        'label_len':tf.train.Feature(int64_list = tf.train.Int64List(value = [label_len])),
                        'hei':tf.train.Feature(int64_list = tf.train.Int64List(value = [line_hei])),
                        'wid':tf.train.Feature(int64_list = tf.train.Int64List(value = [line_wid])),
                        'label':tf.train.Feature(bytes_list= tf.train.BytesList(value = [line_labels_str])),
                        'image':tf.train.Feature(bytes_list = tf.train.BytesList(value = [line_bytes]))
                    }
                ))
                writer.write(example.SerializeToString())
                writer.close()
if __name__ == '__main__':
    write_record()

import tensorflow as tf
import glob
from collections import defaultdict
from itertools import groupby
#from write_record import write_record_file

sess = tf.Session()
image_filenames = glob.glob("./Images/n02*/*.jpg")
#print (image_filenames[0:2])
##['./Images/n02106662-German_shepherd/n02106662_9226.jpg', 
# './Images/n02106662-German_shepherd/n02106662_24577.jpg']

image_filenames_with_breed = map(lambda filenames : (filenames.split('/')[2],filenames),
    image_filenames)
# list_image = list(image_filenames_with_breed)
# print(list_image[2][0],len(list_image))
# ('n02106662-German_shepherd', './Images/n02106662-German_shepherd/n02106662_16418.jpg') 20580

testing_dataset = defaultdict(list)
training_dataset = defaultdict(list)

for dog_breed, breed_images in groupby(image_filenames_with_breed,lambda x:x[0]):
    #for name, group in groupby(),group 为数据块，在这里是（dog_breed, filenames)结构
    #x[0] n02106662-German_shepherd(即品种)
    for i, breed_image in enumerate(breed_images):
        if i%5 == 0:
            testing_dataset[dog_breed].append(breed_image[1])
            #print(dog_breed,breed_image[1])
            #n02085620-Chihuahua ./Images/n02085620-Chihuahua/n02085620_1916.jpg
        else:
            training_dataset[dog_breed].append(breed_image[1])

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

write_record_file(testing_dataset, "./output/testing-images/testing-image")
write_record_file(training_dataset, "./output/training-images/training-image")

#加载图像
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

record_dataset = tf.data.TFRecordDataset(tf.train.match_filenames_once(
    "./output/training-images/*.tfrecords"))

record_dataset.map(parse_serialized)
record_dataset.shuffle(buffer_size = 10).batch(batch_size = 3)
print(record_dataset)
sess.close()
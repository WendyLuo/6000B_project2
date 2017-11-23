# -*- coding: utf-8 -*-

import tensorflow as tf
import cv2

def raw_to_tfrecords(file, data_root, n_name="data.tfrecords", resize=None):
    writer = tf.python_io.TFRecordWriter(data_root+"/"+n_name)
    num_example = 0
    with open(file, 'r')as f:
        for line in f.readlines():
            line = line.split()
            image = cv2.imread(data_root+'/'+line[0])
            if resize is not None:
                image = cv2.resize(image, resize)

            height, width, nchannel = image.shape
            label = int(line[1])   #get each image's label
            example = tf.train.Example(features=tf.train.Features(feature={'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                'nchannel': tf.train.Feature(int64_list=tf.train.Int64List(value=[nchannel])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))}))
            serialized = example.SerializeToString()
            writer.write(serialized)
            num_example += 1
    print file, "sample volume：", num_example
    writer.close()


def read_from_tfrecords(filename, num_epoch=None):
    filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epoch)

    #define a reader, access the next record
    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)
#parse the current record
    example = tf.parse_single_example(serialized, features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'nchannel': tf.FixedLenFeature([], tf.int64),
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    })

    label = tf.cast(example['label'], tf.int32)
    image = tf.decode_raw(example['image'], tf.uint8)
    #restore the image to its original dimensions
    image = tf.reshape(image, tf.stack([
        tf.cast(example['height'], tf.int32),
        tf.cast(example['width'], tf.int32),
        tf.cast(example['nchannel'], tf.int32)]))

    return image, label

def get_batch(image, label, batch_size, crop_size):
    distorted_image = tf.random_crop(image, [crop_size, crop_size, 3])
    distorted_image = tf.image.random_flip_up_down(distorted_image)
    images, label_batch = tf.train.shuffle_batch([distorted_image, label], batch_size=batch_size,
                                                num_threads=16, capacity=50000, min_after_dequeue=10000)

    return images, tf.reshape(label_batch, [batch_size])

#test phase using get_batch function
def test_batch(image, label, batch_size, crop_size):
    distorted_image = tf.image.central_crop(image, 39./45.)
    distorted_image = tf.random_crop(distorted_image, [crop_size, crop_size, 3])  # crop randomly
    images, label_batch = tf.train.batch([distorted_image, label], batch_size=batch_size)

    return images, tf.reshape(label_batch, [batch_size])
























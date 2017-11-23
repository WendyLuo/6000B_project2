# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import glob
import numpy as np
from skimage import io,transform
from tensorCNN import network

file = "/Users/wendyti/PycharmProjects/deepLearning/data/flower_photos/test/"
res = (39, 39)

def read_img(path):
    imgs=[]
    for im in glob.glob(file+'/*.jpg'):
        # print('reading the images:%s'%(im))
        img=io.imread(im)
        img=transform.resize(img, res)
        imgs.append(img)

    return np.asarray(imgs,np.float32)

testImage = read_img(file)

net = network()

sess = tf.Session()
x = tf.placeholder(tf.float32, [None,39,39,3])
y_ = tf.placeholder(tf.float32, [None, 5])

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

saver.restore(sess, '/Users/wendyti/PycharmProjects/deepLearning/Try_model')
graph = tf.get_default_graph()

logits = net.inference(testImage)

result = sess.run(tf.argmax(logits, 1), feed_dict={x: testImage})
print result

try:
    with open("image_label.txt", "wb")as fi:
        for item in result:
            fi.write(str(item))
            fi.write('\n')
    fi.close()
except Exception, e:
    print e
sess.close()



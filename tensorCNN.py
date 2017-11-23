# -*- coding: utf-8 -*-

import tensorflow as tf
from DataProcess import raw_to_tfrecords, read_from_tfrecords, get_batch, test_batch
import cv2
import os

class network(object):
    def __init__(self):
        with tf.variable_scope("weights"):
            self.weights = {
                # 39*39*3->36*36*20->18*18*20        卷积核patch的大小是4*4,3是channel,20个feature map
                'conv1': tf.get_variable('conv1', [4, 4, 3, 20],
                                         initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                # 18*18*20->16*16*40->8*8*40      3*3 20个feature map,对应输出40个feature map
                'conv2': tf.get_variable('conv2', [3, 3, 20, 40],
                                         initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                # 8*8*40->6*6*60->3*3*60
                'conv3': tf.get_variable('conv3', [3, 3, 40, 60],
                                         initializer=tf.contrib.layers.xavier_initializer_conv2d()),
                # 3*3*60->120
                'fc1': tf.get_variable('fc1', [3 * 3 * 60, 120], initializer=tf.contrib.layers.xavier_initializer()),
                # 120->5
                'fc2': tf.get_variable('fc2', [120, 5], initializer=tf.contrib.layers.xavier_initializer()),
            }
        with tf.variable_scope("biases"):
            self.biases = {
                'conv1': tf.get_variable('conv1', [20, ],
                                         initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv2': tf.get_variable('conv2', [40, ],
                                         initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'conv3': tf.get_variable('conv3', [60, ],
                                         initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'fc1': tf.get_variable('fc1', [120, ],
                                       initializer=tf.constant_initializer(value=0.0, dtype=tf.float32)),
                'fc2': tf.get_variable('fc2', [5, ], initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))

            }

    def inference(self, images):
        # 向量转为矩阵
        images = tf.reshape(images, shape=[-1, 39, 39, 3])  # [batch, in_height, in_width, in_channels]
        images = (tf.cast(images, tf.float32) / 255. - 0.5) * 2  # 归一化处理

        # 第一层           strides[0]和strides[3]是默认为1，间两个1代表padding时在x方向运动一步，y方向运动一步
        conv1 = tf.nn.bias_add(tf.nn.conv2d(images, self.weights['conv1'], strides=[1, 1, 1, 1], padding='VALID'),
                               self.biases['conv1'])

        relu1 = tf.nn.relu(conv1)
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # 第二层
        conv2 = tf.nn.bias_add(tf.nn.conv2d(pool1, self.weights['conv2'], strides=[1, 1, 1, 1], padding='VALID'),
                               self.biases['conv2'])
        relu2 = tf.nn.relu(conv2)
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # 第三层
        conv3 = tf.nn.bias_add(tf.nn.conv2d(pool2, self.weights['conv3'], strides=[1, 1, 1, 1], padding='VALID'),
                               self.biases['conv3'])
        relu3 = tf.nn.relu(conv3)
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # 全连接层1，先把特征图转为向量
        flatten = tf.reshape(pool3, [-1, self.weights['fc1'].get_shape().as_list()[0]])

        drop1 = tf.nn.dropout(flatten, 0.5)
        fc1 = tf.matmul(drop1, self.weights['fc1']) + self.biases['fc1']

        fc_relu1 = tf.nn.relu(fc1)

        fc2 = tf.matmul(fc_relu1, self.weights['fc2']) + self.biases['fc2']

        return fc2

    def inference_test(self, images):
        # 向量转为矩阵      3表示channel的数量，RGB为3，黑白的为1
        images = tf.reshape(images, shape=[-1, 39, 39, 3])  # [batch, in_height, in_width, in_channels],-1表示不考虑输入的图片的维度
        images = (tf.cast(images, tf.float32) / 255. - 0.5) * 2  # 归一化处理

        # 第一层
        conv1 = tf.nn.bias_add(tf.nn.conv2d(images, self.weights['conv1'], strides=[1, 1, 1, 1], padding='VALID'),
                               self.biases['conv1'])

        relu1 = tf.nn.relu(conv1)
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # 第二层
        conv2 = tf.nn.bias_add(tf.nn.conv2d(pool1, self.weights['conv2'], strides=[1, 1, 1, 1], padding='VALID'),
                               self.biases['conv2'])
        relu2 = tf.nn.relu(conv2)
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # 第三层
        conv3 = tf.nn.bias_add(tf.nn.conv2d(pool2, self.weights['conv3'], strides=[1, 1, 1, 1], padding='VALID'),
                               self.biases['conv3'])
        relu3 = tf.nn.relu(conv3)
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # 全连接层1，先把特征图转为向量
        flatten = tf.reshape(pool3, [-1, self.weights['fc1'].get_shape().as_list()[0]])

        fc1 = tf.matmul(flatten, self.weights['fc1']) + self.biases['fc1']
        fc_relu1 = tf.nn.relu(fc1)

        fc2 = tf.matmul(fc_relu1, self.weights['fc2']) + self.biases['fc2']

        return fc2

        # calculate softmax_cross_entropy_loss function

    def sorfmax_loss(self, predicts, labels):
        predicts = tf.nn.softmax(predicts)
        labels = tf.one_hot(labels, self.weights['fc2'].get_shape().as_list()[1])
        loss = -tf.reduce_mean(labels * tf.log(predicts))  # tf.nn.softmax_cross_entropy_with_logits(predicts, labels)
        self.cost = loss
        return self.cost

    def optimer(self, loss, lr=0.005):  #learning rate is 0.005
        train_optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

        return train_optimizer

def train():
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    raw_to_tfrecords("data/train.txt", "data", 'train.tfrecords', (45, 45))
    image, label = read_from_tfrecords('data/train.tfrecords')
    batch_image, batch_label = get_batch(image, label, batch_size=50, crop_size=39)  # batch 生成测试 crop_size剪裁图片

    #declare the network
    net = network()
    logits = net.inference(batch_image)

    loss = net.sorfmax_loss(logits, batch_label)
    opti = net.optimer(loss)

    # use validation set to test the accuracy and loss
    raw_to_tfrecords("data/val.txt", "data", 'val.tfrecords', (45, 45))
    test_image, test_label = read_from_tfrecords('data/val.tfrecords', num_epoch=None)
    test_images, test_labels = test_batch(test_image, test_label, batch_size=120, crop_size=39)  # batch 生成测试
    test_inf = net.inference_test(test_images)
    correct_prediction = tf.equal(tf.cast(tf.argmax(test_inf, 1), tf.int32), test_labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.initialize_all_variables()
    with tf.Session() as session:
        session.run(init)
        #declare the variable to save the train model
        saver = tf.train.Saver()
        model_path = os.path.abspath('.')+'/'+'Try_model'
        max_acc = 0

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        max_iter = 80000  # for iteration


        for iter in range(max_iter):
            loss_np, _, label_np, image_np, inf_np = session.run([loss, opti, batch_label, batch_image, logits])

            if iter % 50 == 0:
                print 'trainloss:', loss_np
            if iter % 500 == 0:
                accuracy_np = session.run([accuracy])
                print '***************test accruacy:', accuracy_np, '*******************'
                if accuracy_np > max_acc:
                    max_acc = accuracy_np
                    saver.save(session, model_path)



        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    train()



# -*- coding: utf-8 -*-
import os
# from PIL import Image
from sklearn.model_selection import KFold
import numpy as np
import cv2
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import scipy.misc


def splitImageFile(file_path):
    image_list, label_list = [], []
    f = open(file_path)
    f = f.readlines()
    for line in f:
        image, label = line.strip().split(" ")
        image_list.append(image), label_list.append(label)
    return image_list, label_list

def splitTestFile(file_path):
    ima_list = []
    f = open(file_path)
    f = f.readlines()
    for line in f:
        image = line.strip()
        ima_list.append(image)
    return ima_list

# def cv_train_test_split(train_data, train_label):
#     cv = 5
#     kf = KFold(n_splits=cv, shuffle=True)
#     train_test_fold = kf.split(train_data)
#     print type(train_test_fold)
#
#     return train_test_fold

def loadImageToArray(file_path, data_path, image_size):
    os.chdir(data_path) #改变工作目录的路径
    file_list, label_list = splitImageFile(file_path)

    image_list = []
    for item in file_list:
        item = cv2.imread(data_path + '/' + item)
        # item = load_img(data_path + '/' + item)
        item = cv2.resize(item, (image_size, image_size))
        # item = scipy.misc.imresize(item, (image_size,image_size))
        item = img_to_array(item)    #ndarray类型
        # item = item.astype(np.float32)
        image_list.append(item)
    image_list = np.array(image_list, dtype=float)

    label_list = map(lambda x: np.int32(x), label_list)
    label_list = np.array(label_list)
    current_path = os.getcwd()  #返回当前的工作路径
    print current_path

    os.chdir(current_path)

    return image_list, label_list

def loadTestImageToArray(file_path, data_path, image_size):

    os.chdir(data_path) #改变工作目录的路径
    file_list = splitTestFile(file_path)
    image_list = []
    for item in file_list:
        item = cv2.imread(data_path + '/' + item)
        item = cv2.resize(item, (image_size, image_size))

        item = img_to_array(item)
        image_list.append(item)
    image_list = np.array(image_list, dtype=float)
    current_path = os.getcwd()  #返回当前的工作路径
    os.chdir(current_path)

    return image_list









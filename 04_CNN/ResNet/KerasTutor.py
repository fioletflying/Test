#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @author   : Feifei
# @IDE      : Pycharm
# @file     : KerasTutor.py
# @time     : 2019/6/22 10:36

import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()


X_train = X_train_orig/255.
X_test = X_test_orig/255.

# 转置，X的数据格式保持一致 （600，1）=》（1，600）
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

# print ("number of training examples = " + str(X_train.shape[0]))
# print ("number of test examples = " + str(X_test.shape[0]))
# print ("X_train shape: " + str(X_train.shape))
# print ("Y_train shape: " + str(Y_train_orig.shape))
# print ("X_test shape: " + str(X_test.shape))
# print ("Y_test shape: " + str(Y_test_orig.shape))

def model(input_shpae):
    X_input = Input(input_shpae)

    X = ZeroPadding2D((3,3))(X_input)

    X = Conv2D(32,(7,7),strides=(1,1),name='conv0')(X)
    X = BatchNormalization(axis=3,name='bn0')(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((2,2),name='max_pool')(X)

    X = Flatten()(X)
    X = Dense(1,activation='sigmoid',name='fc')(X)

    model = Model(inputs =  X_input, outputs = X,name='HappyModel')

    return model


def HappyModel(input_shpae):
    X_input = Input(input_shpae)

    X = ZeroPadding2D((3,3))(X_input)

    X = Conv2D(32,(7,7),strides=(1,1),name='conv0')(X)
    X = BatchNormalization(axis=3,name='bn0')(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((2,2),name='max_pool')(X)

    X = Flatten()(X)
    X = Dense(1,activation='sigmoid',name='fc')(X)

    model = Model(inputs =  X_input, outputs = X,name='HappyModel')

    return model



def trainModle():

    happyModel.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])
    happyModel.fit(x=X_train, y=Y_train, epochs=10, batch_size=32)




happyModel = HappyModel(X_train.shape[1:])
#trainModle()
preds = happyModel.evaluate(X_test,Y_test)

print()
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))


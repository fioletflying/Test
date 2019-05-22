#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @author   : Feifei
# @IDE      : Pycharm
# @file     : AlexNet.py
# @time     : 2019/5/16 8:44

from datetime import datetime
import math
import time
import tensorflow as tf


batch_size = 32
num_batches = 100


def print_activations(t):
    print(t.op.name,' ',t.get_shape().as_list())


def inference(images):

    parameters = []

    # 第一层卷积卷积核：size: 11x11 channel:3  depth:64
    # stride: 4x4  padding: 边缘填零

    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11,11,3,64],
                             dtype=tf.float32,stddev = 1e-1),
                             name = 'weights')
        conv = tf.nn.conv2d(images,kernel,[1,4,4,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,shape=[64],dtype=tf.float32),
                             trainable=True,name = 'biases')
        bias = tf.nn.bias_add(conv,biases)
        conv1 = tf.nn.relu(bias,name=scope)
        print_activations(conv1)
        parameters += [kernel, biases]

    # 池化： szie:3x3  stride: 2x2
    lrn1 = tf.nn.lrn(conv1,4,bias = 1.0,alpha=0.001/9,beta=0.75,name='lrn1')
    pool1 = tf.nn.max_pool(lrn1,ksize=[1,3,3,1],strides=[1,2,2,1],
                           padding='VALID',name='pool1')
    print_activations(pool1)

    # 第二层卷积卷积核：size: 5x5 channel:64  depth:193
    # stride: 1x1  padding: 边缘填零: 做完卷积后size不会变

    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5,5,64,192],
                                                 dtype=tf.float32,stddev=1e-1),
                             name = 'weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        print_activations(conv2)
        parameters += [kernel, biases]

    # 池化： szie:3x3  stride: 2x2
    lrn2 = tf.nn.lrn(conv2,4,bias = 1.0,alpha=0.001/9,beta=0.75,name='lrn2')
    pool2 = tf.nn.max_pool(lrn2,ksize=[1,3,3,1],strides=[1,2,2,1],
                           padding='VALID',name='pool2')
    print_activations(pool2)


    # 第三层卷积卷积核：size: 3x3 channel:192  depth:384
    # stride: 1x1  padding: 边缘填零: 做完卷积后size不会变
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,192,384],
                            dtype=tf.float32,stddev=1e-1),
                             name = 'weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        print_activations(conv3)
        parameters += [kernel, biases]

    # 第四层卷积卷积核：size: 3x3 channel:384  depth:256
    # stride: 1x1  padding: 边缘填零: 做完卷积后size不会变
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,384,256],
                            dtype=tf.float32,stddev=1e-1),
                             name = 'weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        print_activations(conv4)
        parameters += [kernel, biases]

    # 第五层卷积卷积核：size: 3x3 channel:256  depth:256
    # stride: 1x1  padding: 边缘填零: 做完卷积后size不会变
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,256,256],
                            dtype=tf.float32,stddev=1e-1),
                             name = 'weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        print_activations(conv5)
        parameters += [kernel, biases]

    # 池化： szie:3x3  stride: 2x2
    pool5 = tf.nn.max_pool(conv5,ksize=[1,3,3,1],strides=[1,2,2,1],
                           padding='VALID',name='pool5')
    print_activations(pool5)

    # 还需要添加相关的FC层4096，4096，1000

    # 全连接层1： FC1 4096
    with tf.name_scope('fcl1') as scope:
        weight = tf.Variable(tf.truncated_normal([6*6*256,4096],stddev=0.1),name = 'weights')
        biases = tf.Variable(tf.constant(0.0,shape=[4096],dtype=tf.float32),trainable=True,name='biases')
        h_pool5_flat = tf.reshape(pool5,[-1,6*6*256])
        fcl1 = tf.nn.relu(tf.matmul(h_pool5_flat,weight)+biases,name=scope)
        drop1 = tf.nn.dropout(fcl1,0.7)
        parameters +=[weight,biases]
        print_activations(fcl1)


    # 全连接层2： FC2 4096
    with tf.name_scope('fcl2') as scope:
        weight = tf.Variable(tf.truncated_normal([4096,4096],stddev=0.1),name = 'weights')
        biases = tf.Variable(tf.constant(0.0,shape=[4096],dtype=tf.float32),trainable=True,name='biases')
        fcl2 = tf.nn.relu(tf.matmul(drop1,weight)+biases,name=scope)
        drop2 = tf.nn.dropout(fcl2,0.7)
        parameters +=[weight,biases]
        print_activations(fcl2)



    # 全连接层3： FC3 100
    with tf.name_scope('fcl3') as scope:
        weight = tf.Variable(tf.truncated_normal([4096,1000],stddev=0.1),name = 'weights')
        biases = tf.Variable(tf.constant(0.0,shape=[1000],dtype=tf.float32),trainable=True,name='biases')
        fcl3 = tf.nn.relu(tf.matmul(drop2,weight)+biases,name=scope)
        parameters +=[weight,biases]
        print_activations(fcl3)


    return fcl3,parameters



'''
session:    Tensorflow Session
target:     需要测评的运算的操作
info_string：名称
'''
def time_tensorflow_run(session,target,info_string):
    num_steps_burn_in = 10
    total_duraion = 0.0
    total_duraion_squared = 0.0

    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s:step %d,duration = %.3f'%
                      (datetime.now(),i-num_steps_burn_in,duration))
                total_duraion += duration
                total_duraion_squared += duration * duration
    mn = total_duraion / num_batches
    vr = total_duraion_squared / num_batches-mn*mn
    sd = math.sqrt(vr)
    print('%s:%s across %d steps, %.3f +/- %.3f sec / batch'%
          (datetime.now(),info_string,num_batches,mn,sd))



def run_benchmark():

    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size,
                                               image_size,
                                               image_size,3],
                                              dtype=tf.float32,
                                              stddev=1e-1))
        pool5,parameters = inference(images)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        time_tensorflow_run(sess,pool5,"Forward")
        objective = tf.nn.l2_loss(pool5)
        grad = tf.gradients(objective,parameters)
        time_tensorflow_run(sess,grad,"Forward-backward")

run_benchmark()










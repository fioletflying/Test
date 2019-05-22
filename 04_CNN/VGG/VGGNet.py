#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @author   : Feifei
# @IDE      : Pycharm
# @file     : VGGNet.py
# @time     : 2019/5/22 16:27
# @info     : 实现VGG16的版本

from datetime import datetime
import math
import time
import tensorflow as tf


def conv_op(input_op,name,kh,kw,n_out,dh,dw,p):

   '''

   :param input_op: 输入的tensor
   :param name: 本层的名字
   :param kh:    kernel 的heiht
   :param kw:    kernel 的width
   :param n_out: 输出的通道数
   :param dh:    步长的高
   :param dw:    步长的宽
   :param p:     参数列表
   :return:

   '''
   # 获取输入通道数
   n_in = input_op.get_shape()[-1].value

   # 在这个命名范围开始创建
   with tf.name_scope(name) as scope:
       kernel = tf.get_variable(scope + "w",
                                shape = [kh,kw,n_in,n_out],
                                dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer_conv2d())
       conv = tf.nn.conv2d(input_op,kernel,(1,dh,dw,1),padding='SAME')
       bias_init_val = tf.constant(0.0,shape=[n_out],dtype = tf.float32)
       biases = tf.Variable(bias_init_val,trainable=True,name='b')
       z = tf.nn.bias_add(conv,biases)
       activation = tf.nn.relu(z,name=scope)
       p+= [kernel,biases]
       return activation


def fc_op(input_op,name,n_out,p):
    '''

    :param input_op: 输入的tensor
    :param name: 本层的名字
    :param n_out: 输出的通道数
    :param p: 参数列表
    :return:
    '''

    # 获取输入通道数
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w",
                                 shape=[n_in,n_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1,shape=[n_out],
                                         dtype=tf.float32),name ='b')
        activation = tf.nn.relu_layer(input_op,kernel,biases,name = scope)
        p+=[kernel,biases]
        return  activation



def mpool_op(input_op,name,kh,k2,dh,dw):
    return tf.nn.max_pool(input_op,
                          ksize=[1,kh,kw,1],
                          strides=[1,dh,dw,1],
                          padding='SAMW',
                          name=name)













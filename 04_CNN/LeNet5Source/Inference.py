#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @author   : Feifei
# @IDE      : Pycharm
# @file     : Inference.py
# @time     : 2019/4/29 11:44

import tensorflow as tf



# 参数变量的定义
# 输入的节点数量，等于图片的像素
INPUT_NODE = 784
# 输出的节点数（0-9）10个数字
OUTPUT_NODE = 10

# 配置神经网络的参数
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

#第一层卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 5

#第二层卷积层的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 5

#全连接层的节点个数
FC_SIZE = 1024

def weight_variable(shape):












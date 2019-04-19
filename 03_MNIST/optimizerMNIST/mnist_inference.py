# -*- coding:utf-8 -*-

import tensorflow as tf

# 输入的节点数量，等于图片的像素
INPUT_NODE = 784
# 输出的节点数（0-9）10个数字
OUTPUT_NODE = 10
# 隐藏层的网络结构，500个节点
LAYER1_NODE = 500

def get_weight_variable(shape,regularizer):

    weights =tf.get_variable(
        "weights",
        shape,
        initializer = tf.truncated_normal_initializer(stddev = 0.1)
    )

    # 将当前变量的正则化损失加入名字为 losses 的集合
    # 使用了 add to  collection 函数将一个张量加入一个集合
    # 这是自定义的集合，不在 TensorFlow 自动管理的集合列表中
    if regularizer != None:
        tf.add_to_collection('losses',regularizer(weights))
    return weights


# 前向传播
def inference(input_tensor,regularizer):
    with tf.variable_scope('layer1'):
        weights = get_weight_variable(
            [INPUT_NODE,LAYER1_NODE],
            regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE],
                                 initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2'):
        weights = get_weight_variable(
            [LAYER1_NODE,OUTPUT_NODE],
            regularizer
        )
        biases = tf.get_variable("biases", [OUTPUT_NODE],
                                 initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2




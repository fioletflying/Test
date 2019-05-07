# -*- coding:utf-8 -*-

import os


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


import mnist_inference

# 一个batch 中的训练数据的个数，
BATCH_SIZE = 100
# 基础的学习率
LEARNING_RATE_BASE = 0.8
# 学习率的衰减
LEARNING_RATE_DECAY = 0.99
# 正则化项的损失函数中的系数
REGULARIZATION_RATE = 0.0001
# 训练步数
TRAINING_STEPS = 30000
# 滑动平均衰减率
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "/model/"
MODEL_NAME = "model.ckpt"


def train(mnist):
    # 输入数据
    x = tf.placeholder(
        tf.float32,
        [BATCH_SIZE, mnist_inference.IMAGE_SIZE,mnist_inference.IMAGE_SIZE,mnist_inference.NUM_CHANNELS],
        name='x-input')

    y_ = tf.placeholder(
        tf.float32,
        [None, mnist_inference.OUTPUT_NODE],
        name='y-input')

    # 计算L2 正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    # 计算前向传播的结果
    y = mnist_inference.inference(x,train='train',regularizer = regularizer)

    # 定义存储轮数的变量，指定不可训练的变量
    global_step = tf.Variable(0, trainable=False)

    # 给定滑动平均衰减率和训练轮数的变量
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    #
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    # 滑动平均之后的前向传播结果
    #average_y = inference(x, variable_averages, weights1, biase1, weights2, biases2)

    # 计算交叉熵
    corss_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # 计算当前batch中所有样例的交叉熵的平均值
    cross_entropy_mean = tf.reduce_mean(corss_entropy)

    # 总损失需要加上正则化的损失
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # 设置衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,  # 基础学习率
        global_step,  # 当前迭代的轮数
        mnist.train.num_examples / BATCH_SIZE,  # 需要迭代的次数
        LEARNING_RATE_DECAY  # 学习率衰减的速度
    )

    # 优化算法来优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    #＃在训练、神经网络模型时，每过一遍数据既需要通过反向传播来更新神经网络中的参数，
    #又要更新每一个参数的滑动平均值，tf.control_dependencies 和 tf.group 两种机制。
    #  train_op  =  tf.group(train_step,  variables_averages_op)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')


    correct_prediction = tf.equal(tf.arg_max(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    # 模型保存
    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):

            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs,(
                BATCH_SIZE,
                mnist_inference.IMAGE_SIZE,
                mnist_inference.IMAGE_SIZE,
                mnist_inference.NUM_CHANNELS
            ))
            _,loss_value,step = sess.run([train_op,loss,global_step],
                                         feed_dict={x:reshaped_xs, y_:ys})

            if i % 1000 == 0:
                saver.save(
                    sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),
                    global_step = global_step
                )
                print("%d steps, validation acc" "using average model is %g" % (step, loss_value))


            sess.run(train_op, feed_dict={x: reshaped_xs, y_: ys})


def main(argv=None):
    mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()

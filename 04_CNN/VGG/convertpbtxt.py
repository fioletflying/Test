#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @author   : Feifei
# @IDE      : Pycharm
# @file     : convertpbtxt.py
# @time     : 2019/5/28 15:13


import tensorflow as tf
from tensorflow.python.platform import gfile
from google.protobuf import text_format


def convert_pb_to_pbtxt(filename):
    with gfile.FastGFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()

        graph_def.ParseFromString(f.read())

        tf.import_graph_def(graph_def, name='')

        tf.train.write_graph(graph_def, './', 'protobuf.pbtxt', as_text=True)
    return


filename = "F:/Amodel/models-master/research/object_detection/panel_mobile_inference_graph/frozen_inference_graph.pb"
convert_pb_to_pbtxt(filename)

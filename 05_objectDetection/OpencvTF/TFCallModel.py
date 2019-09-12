#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @author   : Feifei
# @IDE      : Pycharm
# @file     : TFCallModel.py
# @time     : 2019/7/25 9:20
import os
import numpy as np
import tensorflow as tf
import cv2 as cv
import time as tm



TEST_IMAGE_PATHS = []
#PATH_TO_TEST_IMAGES_DIR ='testImgs'
PATH_TO_TEST_IMAGES_DIR ='testImgsspot'
for fpathe, dirs, fs in os.walk(PATH_TO_TEST_IMAGES_DIR):
    for f in fs:
        if os.path.splitext(f)[1] == '.jpg' or os.path.splitext(f)[1] == '.png':
            TEST_IMAGE_PATHS.append(os.path.join(fpathe, f))




# Read the graph.
with tf.gfile.FastGFile('frozen_inference_graph_faster_spot.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())



with tf.Session() as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    for img_path in TEST_IMAGE_PATHS:
        # Read and preprocess an image.
        start_time = tm.time()
        img = cv.imread(img_path)
        rows = img.shape[0]
        cols = img.shape[1]
        inp = cv.resize(img, (300, 300))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

        # Run the model
        out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                        sess.graph.get_tensor_by_name('detection_scores:0'),
                        sess.graph.get_tensor_by_name('detection_boxes:0'),
                        sess.graph.get_tensor_by_name('detection_classes:0')],
                       feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

        # Visualize detected bounding boxes.
        num_detections = int(out[0][0])
        for i in range(num_detections):
            classId = int(out[3][0][i])
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]
            textInfo = str(classId) + ":" + str(score)
            if score > 0.3:
                x = bbox[1] * cols
                y = bbox[0] * rows
                right = bbox[3] * cols
                bottom = bbox[2] * rows
                cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
                cv.putText(img,textInfo,(int(x),int(y)), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0));
        end_time = tm.time()
        timeInfo = str(end_time - start_time)+"s"
        cv.imshow(timeInfo, img)
        print(timeInfo)
        c = cv.waitKey(0)
        if c =='n':
            break



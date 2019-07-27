#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @author   : Feifei
# @IDE      : Pycharm
# @file     : CVCallModel.py
# @time     : 2019/7/25 9:22


import cv2 as cv
import time as tm
import os

#print (cv.getBuildInformation())
TEST_IMAGE_PATHS = []
PATH_TO_TEST_IMAGES_DIR ='testImgs'
for fpathe, dirs, fs in os.walk(PATH_TO_TEST_IMAGES_DIR):
    for f in fs:
        if os.path.splitext(f)[1] == '.jpg' or os.path.splitext(f)[1] == '.png':
            TEST_IMAGE_PATHS.append(os.path.join(fpathe, f))

cvNet = cv.dnn.readNetFromTensorflow('frozen_inference_graph_faster.pb', 'graph_faster.pbtxt')

for path_img in TEST_IMAGE_PATHS:

    start_time = tm.time()
    img = cv.imread(path_img)
    rows = img.shape[0]
    cols = img.shape[1]


    #width = height = 300
   # image = cv.resize(img, ((int(cols * height / rows), width)))
    #img = image[0:height, image.shape[1] - width:image.shape[1]]
    #cvNet.setInput(cv.dnn.blobFromImage(img, 1.0 / 127.5, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False))
   # cvOut = cvNet.forward()


    #cvNet.setInput(cv.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
    cvNet.setInput(cv.dnn.blobFromImage(img,size=(600, 1024),swapRB=True, crop=False))
    cvOut = cvNet.forward()

    for detection in cvOut[0,0,:,:]:
        score = float(detection[2])
        textInfo = str(score)
        if score > 0.3:
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
            cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
            #cv.putText(img, textInfo, (int(left), int(top)), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255));
    end_time = tm.time()
    info_time = str(end_time - start_time) + "s"
    print(info_time)
    cv.imshow('img', img)
    c = cv.waitKey()
    if c == 'n':
        break
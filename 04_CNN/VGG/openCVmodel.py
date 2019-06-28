import cv2 as cv

weights = "F:/Amodel/models-master/research/object_detection/panel_mobile_inference_graph/frozen_inference_graph.pb"
pbtxt = "F:/Amodel/models-master/research/object_detection/panel_mobile_inference_graph/testPanel.pbtxt"

cvNet = cv.dnn.readNetFromTensorflow(weights, pbtxt)

img = cv.imread('F:/Study/Opencv/dnn/ssdModel/ssdModel/20190525_102902.jpg')
rows = img.shape[0]
cols = img.shape[1]
cvNet.setInput(cv.dnn.blobFromImage(img, size=(300, 300), mean=(127.5,127.5,127.5), swapRB=True, crop=False))
cvOut = cvNet.forward()

for detection in cvOut[0,0,:,:]:
    score = float(detection[2])
    if score > 0.7:
        left = detection[3] * cols
        top = detection[4] * rows
        right = detection[5] * cols
        bottom = detection[6] * rows
        cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)

cv.imshow('img', img)
cv.waitKey()
import numpy as np
import math
import cv2
import os


def recog(img_ftp):
    image = cv2.imread(img_ftp)
    image = cv2.transpose(image)

    detector = cv2.CascadeClassifier("./utils/haarcascade_frontalface_default.xml")
    rects = detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=2, minSize=(10, 10),
                                      flags=cv2.CASCADE_SCALE_IMAGE)
    tl=[]
    for (x,y,w,h) in rects:
        tl = image[y:y + w, x:x + h]
    if len(tl) == 0:
        print('cannot dectect the face')
        img_finish=tl
    else:
        img_finish = cv2.resize(tl, (112, 112))

    return img_finish

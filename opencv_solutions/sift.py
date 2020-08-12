# -*- coding: utf-8 -*-

import numpy as np
import cv2 as cv
img = cv.imread('/home/mahdi/Pictures/test/1.jpeg')
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
sift = cv.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)
img=cv.drawKeypoints(gray,kp,img)
cv.imwrite('sift_keypoints.jpg',img)
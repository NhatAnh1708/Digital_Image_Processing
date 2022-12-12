import cv2 as cv
import numpy as np

img_path="../final4.jpg"


img=cv.imread(img_path)
kernel = np.ones((3,3),np.uint8)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

dilation = cv.dilate(gray,kernel,iterations = 1)
cv.imshow('out',dilation)
cv.waitKey(0)
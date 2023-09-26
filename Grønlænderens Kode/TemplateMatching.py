import cv2 as cv
import numpy as np
import math

#defining the input image
input = cv.imread("C:/Users/minik/Desktop/VSCode/neonText.png")

#converting to grayscale
img_gray = cv.cvtColor(input, cv.COLOR_BGR2GRAY)

#defining the template
template = cv.imread("C:/Users/minik/Desktop/VSCode/heart.png", 0)

#store width and height of template in w and h
w, h = template.shape[::-1]

#perform match operations
res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)

#specify a threshold
threshold = 0.8

output = res.copy()

for y in range(res.shape[0]):
    for x in range(res.shape[1]):
        if res[y,x] < threshold:
            output[y,x] = 0
        else:
            output[y,x] = 255

#show the final image with the matched area
cv.imshow('input',input)
cv.imshow('res',res)
cv.imshow('output',output)
cv.waitKey(0)   




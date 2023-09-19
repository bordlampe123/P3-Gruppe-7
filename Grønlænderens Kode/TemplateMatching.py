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

ThresRes = cv.threshold(res, threshold, 255, cv.THRESH_BINARY)[1]

output = input.copy()

ThresRes = (ThresRes).astype(np.uint8) 

#show the final image with the matched area

cv.imshow('input',input)
cv.imshow('res',res)
cv.imshow('ThresRes',ThresRes)
cv.waitKey(0)   




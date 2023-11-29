import cv2 as cv
import numpy as np
import math

#defining the input image
input = cv.imread("C:/Users/minik/Desktop/VSCode/55.jpg", cv.IMREAD_COLOR)

#converting to grayscale
img_gray = cv.cvtColor(input, cv.COLOR_BGR2GRAY)

#defining the template
template = cv.imread("C:/Users/minik/Desktop/VSCode/CrownTemplate.png", cv.IMREAD_COLOR)
template_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

img_HSV = cv.cvtColor(template, cv.COLOR_BGR2HSV)

w, h = template_gray.shape[::-1]

#perform match operations
res = cv.matchTemplate(input, template, cv.TM_CCOEFF)

#specify a threshold
threshold = 0.1

#loc = np.where(res >= threshold)
#for pt in zip(*loc[::-1]):
   # cv.rectangle(input, pt, (pt[0]+w, pt[1]+h), (0,0,255), 2)

#cv.imwrite('res.png',input)

output = res.copy()

for y in range(res.shape[0]):
    for x in range(res.shape[1]):
        if res[y,x] < threshold:
            output[y,x] = 0
        else:
            output[y,x] = 255

#show the final image with the matched area
cv.imshow('input',input)
cv.imshow('img_HSV',img_HSV)
cv.imshow('img_gray',img_gray)
cv.imshow('res',res)
cv.imshow('template',template)
cv.imshow('output',output)
cv.waitKey(0)   
cv.destroyAllWindows()




import cv2 as cv
import numpy as np
import math

#defining the path to the image
input = cv.imread("C:/Users/minik/Desktop/VSCode/hand-eye-calibration/7.jpg")

red = input[:,:,2]
pen = cv.threshold(red, 220, 255, 0)

ys, xs = np.where(pen[1] == 255)
print(xs, ys)

y = int(np.mean(ys))
x = int(np.mean(xs))

print(x, y)

input = cv.circle(input, (x, y), 10, (0, 255, 0), -1)

cv.imshow("input", input)
cv.imshow("pen", pen[1])
cv.waitKey(0)
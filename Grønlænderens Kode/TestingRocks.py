import cv2 as cv
import numpy as np

# Load the image
image = cv.imread("C:/Users/minik/Desktop/VSCode/rocks.jpg")
image = cv.resize(image, (0, 0), fx=0.2, fy=0.2)
cv.imshow("Image", image)

HSVImage = cv.cvtColor(image, cv.COLOR_BGR2HSV)
H, S, V = cv.split(HSVImage)

cv.imshow("H", H)
cv.imshow("S", S)
cv.imshow("V", V)

VHSUB = cv.subtract(V, H)
cv.imshow("VHSUB", VHSUB)

VHH = cv.subtract(H, VHSUB)
cv.imshow("VHH", VHH)

cv.imshow("HSVImage", HSVImage)
cv.waitKey(0)
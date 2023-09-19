import cv2

import numpy as np

img = cv2.imread("Pierres Mappe/lion.jpg")
img_gray = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
img_gray_enhan = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
img_temp = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)

def NestedMethod(A, B):
    for y in range(A.shape[0]):
        for x in range(A.shape[1]):
            B[y, x] = (int(A[y,x,0]) + int(A[y,x,1]) + int(A[y,x,2]))/3

def GrayEnhan(A, B, a, b):

    temp = []

    for y in range(A.shape[0]):
        for x in range(A.shape[0]):
            temp.append(A[y,x,0])

    minVal = min(temp)
    maxVal = max(temp)
    
    leftSpan = maxVal - minVal

    for y in range(A.shape[0]):
        for x in range(A.shape[0]):
            img_temp = A[y,x,0] + a
            B[y, x] = img_temp


NestedMethod(img, img_gray) 
GrayEnhan(img_gray, img_gray_enhan, 1, 0)
img_gray_cv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("Billedet", img)
cv2.imshow("GrayScale DIY", img_gray)
cv2.imshow("GrayScaleCV", img_gray_cv)
cv2.imshow("4", img_gray_enhan)

cv2.waitKey(0)
                 
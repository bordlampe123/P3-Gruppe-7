import cv2

from PIL import Image

import numpy as np

input = np.array([[100, 10, 110, 11],
                 [9, 50, 8, 49],
                 [105, 12, 112, 9],
                 [14, 52, 15, 54]])

RGB = np.array([[[1, 0, 0], [0, 0, 2], [0, 0, 3]],
               [[5, 0, 0], [0, 6, 0], [0, 7, 0]],
               [[0, 9, 0], [0, 2, 0], [0, 1, 0]]])

#RGB[række, pixel, RGB]
print(input[1,0])

def RGBb(x, y):
    RGB[y, x, 0] = input[y+1, x+1]
    RGB[y, x, 1] = input[y, x+1]
    RGB[y, x, 2] = input[y, x]

def RGBgb(x, y):
    RGB[y, x, 0] = input[y+1, x]
    RGB[y, x, 1] = input[y, x]
    RGB[y, x, 2] = input[y, x-1]

def RGBgr(x, y):
    RGB[y, x, 0] = input[y, x+1]
    RGB[y, x, 1] = input[y, x]
    RGB[y, x, 2] = input[y-1, x]

def RGBr(x, y):
    RGB[y, x, 0] = input[y, x]
    RGB[y, x, 1] = input[y, x-1]
    RGB[y, x, 2] = input[y-1, x-1]



RGBb(0,0)


for y in range(3):
    for x in range(3):
        if (y % 2) == 0:
            if (x % 2) == 0:
                RGBb(x, y)
            else:
                RGBgb(x, y)
        else:
            if (x % 2) ==0:
                RGBgr(x, y)
            else:
                RGBr(x, y)

print(RGB[0])
print(RGB[1])
print(RGB[2])

blank_image = np.zeros((3,3,3), np.uint8)

for y in range(3):
    for x in range(3):
        blank_image[y, x] = RGB[y, x]

correct = cv2.cvtColor(blank_image, cv2.COLOR_RGB2BGR)

cv2.imshow("Our window", correct)
cv2.waitKey(0)
cv2.imwrite("Output2.png", correct)



""" image = cv2.imread("C:/Users/pierr/Desktop/P3-Gruppe-7/Pierres Mappe/finger.jpg", cv2.IMREAD_GRAYSCALE)

for y in range(image.shape[0]):
    for x in range(image.shape[1]):
        print(f"Pixel value with range at ({x}, {y}): {image[y, x]}") """


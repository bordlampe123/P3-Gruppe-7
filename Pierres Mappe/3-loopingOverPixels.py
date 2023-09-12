import cv2

import numpy as np

input = np.array([100, 10, 110, 11],
                 [9, 50, 8, 49],
                 [105, 12, 112, 9],
                 [14, 52, 15, 54])

RGB = np.array([[1, 0, 0], [0, 0, 2], [0, 0, 3], [0, 0, 4]],
               [[5, 0, 0], [0, 6, 0], [0, 7, 0], [0, 8, 0]],
               [[0, 9, 0], [0, 2, 0], [0, 1, 0], [0, 4, 0]],
               [[3, 0, 0], [6, 0, 0], [8, 0, 0], [9, 0, 0]])


print(RGB[0,0,0])

""" image = cv2.imread("C:/Users/pierr/Desktop/P3-Gruppe-7/Pierres Mappe/finger.jpg", cv2.IMREAD_GRAYSCALE)

for y in range(image.shape[0]):
    for x in range(image.shape[1]):
        print(f"Pixel value with range at ({x}, {y}): {image[y, x]}") """


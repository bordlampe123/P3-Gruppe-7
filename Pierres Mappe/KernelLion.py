import cv2
import numpy as np

img = cv2.imread("Pierres Mappe/lion.jpg")
img_out = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

def MeanFilter(A, B, z):

    kernel = np.ones((z*2+1, z*2+1), np.uint8)

    for y in range(z, img.shape[0]-z):
        starty = y-z    
        for x in range(z, img.shape[1]-z):
            startx = x-z

            sum = np.zeros(3)

            for yy in range(kernel.shape[0]):
                for xx in range(kernel.shape[1]):
                    sum = np.add(sum, A[starty+yy, startx+xx])

            sumnorm = np.divide(sum, kernel.shape[0]**2)        

            B[y, x] = sumnorm

MeanFilter(img, img_out, 5)

cv2.imshow("Billedet", img)
cv2.imshow("Filtered", img_out)
cv2.waitKey(0)
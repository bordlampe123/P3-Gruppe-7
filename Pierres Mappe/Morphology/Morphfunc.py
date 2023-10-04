import cv2
import numpy as np

img = cv2.imread("Pierres Mappe/Morphology/dotsBin.png")
img_out = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
img_out2 = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
img_out3 = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
img_out4 = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)


def Erosion(A, B, z):

    kernel = np.empty((z*2+1, z*2+1), np.uint8)
    kernel.fill(255)   

    for y in range(z, img.shape[0]-z):
        starty = y-z    
        for x in range(z, img.shape[1]-z):
            startx = x-z
            zeros = 0

            for yy in range(kernel.shape[0]):
                for xx in range(kernel.shape[1]):
                    if A[starty+yy, startx+xx, 0] != kernel[yy, xx]:
                        zeros += 1
                        

            if zeros != 0:
                B[y, x] = 2
            else:
                B[y, x] = 255

def Dilution(A, B, z):

    kernel = np.empty((z*2+1, z*2+1), np.uint8)
    kernel.fill(255)     

    for y in range(z, img.shape[0]-z):
        starty = y-z    
        for x in range(z, img.shape[1]-z):
            startx = x-z
            zeros = 0

            for yy in range(kernel.shape[0]):
                for xx in range(kernel.shape[1]):
                    if A[starty+yy, startx+xx, 0] == kernel[yy, xx]:
                        zeros += 1
                        

            if zeros > 0:
                B[y, x] = 255
            else:
                B[y, x] = 0

def Opening(A, B, z):
    img_temp = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    Erosion(A, img_temp, z)
    Dilution(img_temp, B, z)

def Closing(A, B, z):
    img_temp = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    Dilution(A, img_temp, z)
    Erosion(img_temp, B, z)


Erosion(img, img_out, 4)
Dilution(img, img_out2, 1)

Opening(img, img_out3, 1)
Closing(img, img_out4, 1)



cv2.imshow("Billedet", img)
cv2.imshow("Erode", img_out)
cv2.imshow("Dilute", img_out2)
cv2.imshow("Open", img_out3)
cv2.imshow("Close", img_out4)



cv2.waitKey(0)
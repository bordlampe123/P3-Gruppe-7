import cv2
import numpy as np

img = cv2.imread("Pierres Mappe/Project Marker Marker/hand-eye-calibration/marker5.jpg")
img = cv2.resize(img, (960, 540))

img_out = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
img_outB = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
img_outG = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
img_outR = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
img_out2 = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
img_out3 = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
img_out4 = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
img_out5 = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)

def Split(A, B, G, R):
    for y in range(A.shape[0]):
        for x in range(A.shape[1]):
            B[y, x] = A[y, x, 0]
            G[y, x] = A[y, x, 1]
            R[y, x] = A[y, x, 2]

def MeanFilter(A, B, z):

    kernel = np.ones((z*2+1, z*2+1), np.uint8)

    for y in range(z, img.shape[0]-z):
        starty = y-z    
        for x in range(z, img.shape[1]-z):
            startx = x-z

            sum = 0

            for yy in range(kernel.shape[0]):
                for xx in range(kernel.shape[1]):
                    sum += A[starty+yy, startx+xx]

            sumnorm = sum/(z*2+1)**2

            B[y, x] = sumnorm

def Thresh(A, B, z):
    for y in range(A.shape[0]):
        for x in range(A.shape[1]):
            if A[y, x] < z:
                B[y, x] = 0
            else:
                B[y, x] = 255

def MinMax(A):

    yMaxIndex = 0

    xMaxIndex = 0

    for y in range(A.shape[0]):
        for x in range(A.shape[1]):
            if A[y ,x] == 255:
                if y > yMaxIndex:
                    yMaxIndex = y
                if x > xMaxIndex:
                    xMaxIndex = x
    print(yMaxIndex, xMaxIndex)
    return xMaxIndex, yMaxIndex

def MinMinMax(A):

    yMaxIndex = 0
    xMin = 0

    for y in range(A.shape[0]):
        for x in range(A.shape[1]):
            if A[y ,x] == 255:
                if y > yMaxIndex:
                    yMaxIndex = y
    for x in range(A.shape[1]):
        if A[yMaxIndex, x] == 255:
            xMin = x
            return xMin, yMaxIndex

    return xMin, yMaxIndex

Split(img, img_outB, img_outG, img_outR)

img_out2 = cv2.subtract(img_outR, img_outG)
img_out3 = cv2.subtract(img_out2, img_outB)

MeanFilter(img_out3, img_out4, 2)

Thresh(img_out4, img_out5, 110)

M = cv2.moments(img_out5)

cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])

pcX, pcY = MinMax(img_out5)
pccX, pccY = MinMinMax(img_out5)

#cv2.circle(img, (cX, cY+20), 2, (255, 255, 255), -1)
cv2.circle(img, (pcX, pcY), 2, (255, 255, 255), -1)
cv2.circle(img, (pccX, pccY), 2, (255, 255, 255), -1)

cv2.circle(img_out5, (cX, cY), 2, (0, 0, 0), -1)
cv2.circle(img_out5, (pcX, pcY), 2, (200, 0, 0), -1)
cv2.circle(img_out5, (pccX, pccY), 2, (200, 0, 0), -1)





cv2.imshow("Threshed", img_out5)
""" cv2.imshow("Smoothed", img_out4)
cv2.imshow("Sub Red-Blue-Green", img_out3)
cv2.imshow("Sub Red-Blue", img_out2)
cv2.imshow("Only Blue", img_outB)
cv2.imshow("Only Green", img_outG)
cv2.imshow("Only Red", img_outR) """
cv2.imshow("Original", img)
cv2.waitKey(0)
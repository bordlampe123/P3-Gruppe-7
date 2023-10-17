import cv2
import numpy as np

img = cv2.imread("C:/Users/stron/PycharmProjects/P3-Gruppe-7/Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Cropped and perspective corrected boards/10.jpg")
img = cv2.resize(img, (960, 540))

img_out = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
img_outB = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
img_outG = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
img_outR = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
img_out2 = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
img_out3 = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)

def Split(A, B, G, R):
    for y in range(A.shape[0]):
        for x in range(A.shape[1]):
            B[y, x] = A[y, x, 0]
            G[y, x] = A[y, x, 1]
            R[y, x] = A[y, x, 2]


Split(img, img_outB, img_outG, img_outR)

img_out2 = cv2.subtract(img_outR, img_outG)
img_out3 = cv2.subtract(img_out2, img_outB)



cv2.imshow("Sub Red-Blue-Green", img_out3)
cv2.imshow("Sub Red-Blue", img_out2)
cv2.imshow("Only Blue", img_outB)
cv2.imshow("Only Green", img_outG)
cv2.imshow("Only Red", img_outR)
cv2.imshow("Original", img)
cv2.waitKey(0)
import cv2
import numpy as np
import random as rd

img = cv2.imread("Pierres Mappe/GrassFire/shapes.png", cv2.IMREAD_GRAYSCALE)
img_threshed = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
img_out2 = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
img_out3 = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
id = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

current_id = 50

def increment():
    global current_id
    current_id += 10

def Thresh(A, B, z):
    for y in range(A.shape[0]):
        for x in range(A.shape[1]):
            if A[y, x] < z:
                B[y, x] = 0
            else:
                B[y, x] = 255

def ignite(y, x, A):
    
    queue = []

    if A[y, x, 0] == 255 and id[y, x, 0] == 0:
        queue.append([y, x])
    
        while len(queue) > 0:

            temp = queue.pop()

            posy = temp[0]
            posx = temp[1]

            id[posy, posx, 0] = current_id

            if A[posy-1, posx, 0] == 255 and id[posy-1, posx, 0] == 0:
                queue.append([posy-1, posx])
            
            if A[posy, posx-1, 0] == 255 and id[posy, posx-1, 0] == 0:
                queue.append([posy, posx-1])

            if A[posy+1, posx, 0] == 255 and id[posy+1, posx, 0] == 0:
                queue.append([posy+1, posx])

            if A[posy, posx+1, 0] == 255 and id[posy, posx+1, 0] == 0:
                queue.append([posy, posx+1])
        
        increment()

Thresh(img, img_threshed, 255)

for y in range(img_threshed.shape[0]):
    for x in range(img_threshed.shape[1]):
        ignite(y, x, img_threshed)
        #print(current_id)

cv2.imshow("Original", img)
cv2.imshow("Threshed", img_threshed)
cv2.imshow("Blobs?", id)
cv2.imshow("wow?", img_out2)
cv2.waitKey(0)
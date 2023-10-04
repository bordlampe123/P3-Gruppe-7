import cv2
import numpy as np
import time

img = cv2.imread("Pierres Mappe/GrassFire/shapes.png", cv2.IMREAD_GRAYSCALE)
img_threshed = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
img_out2 = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
img_out3 = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
id = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

count = 0

current_id = 0

def Thresh(A, B, z):
    for y in range(A.shape[0]):
        for x in range(A.shape[1]):
            if A[y, x] < z:
                B[y, x] = 0
            else:
                B[y, x] = 255

def ignite(y, x, A):

    #print(f"x: {x}, y: {y}, val: {img[y,x]}")
    #print("hej igen")
    queue = []

    if id[y, x, 0] == 0 and A[y, x] == 255:
        print(y, x)
        print(current_id)
        id[y, x, 0] = current_id

        if id[y-1, x, 0] == 0 and A[y-1, x] == 255:
            queue.append([y-1, x])
        if id[y, x-1, 0] == 0 and A[y, x-1] == 255:
            queue.append([y, x-1])
        if id[y+1, x, 0] == 0 and A[y+1, x] == 255:
            queue.append([y+1, x])
        if id[y, x+1, 0] == 0 and A[y, x+1] == 255:
            queue.append([y, x+1])
        print(queue)
        #time.sleep(0.5)
        if not queue:
            print("Damn")
            time.sleep(2)
            return
        #print(queue)
        temp = queue.pop()
        #print(queue)
        #time.sleep(0.05)
        ignite(temp[0], temp[1], A)
        #print("done?")        
        
Thresh(img, img_threshed, 255) 

for y in range(img_threshed.shape[0]):
        for x in range(img_threshed.shape[1]):
            #print(f"wow: {y}, {x}")
            if id[y, x, 0] == 0 and img_threshed[y, x] == 255:
                current_id += 1
                ignite(y, x, img_threshed)
            

print(count)

bgr = cv2.cvtColor(id, cv2.COLOR_HSV2BGR)

cv2.imshow("Fuck", img)
cv2.imshow("Original", img_threshed)
cv2.imshow("Blobs", bgr)
cv2.waitKey(0)
import cv2
import numpy as np
import statistics as st

img = cv2.imread("Pierres Mappe/Kingdomino/Billeder/1.jpg")

img_blurred = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
img_mean = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
hsv_mod = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)


img_outB = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
img_outG = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
img_outR = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
img_MeanGray = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
img_MeanGrayBlurred = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)

img_Meadows = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
img_Forest = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)

current_id = 1

def increment():
    global current_id
    current_id += 1

def Meanifier(A, B):

    meadfindB = [0]
    meadfindG = [0]
    meadfindR = [0]

    for y in range(5):
        starty = y*100
        #print("y" + str(starty))
        for x in range(5):
            startx = x*100
            #print("x" + str(startx))
            for yy in range(100):
                for xx in range(100):
                    #print(starty+yy, startx+xx)
                    meadfindB.append(A[starty+yy, startx+xx, 0])
                    meadfindG.append(A[starty+yy, startx+xx, 1])
                    meadfindR.append(A[starty+yy, startx+xx, 2])
                    #print(meadfind)
            medianB = st.mean(meadfindB)
            medianG = st.mean(meadfindG)
            medianR = st.mean(meadfindR)
            #print(median)
            for yy in range(100):
                for xx in range(100):
                    B[starty+yy, startx+xx, 0] = medianB
                    B[starty+yy, startx+xx, 1] = medianG
                    B[starty+yy, startx+xx, 2] = medianR
            meadfindB = [0]
            meadfindG = [0]
            meadfindR = [0]

Meanifier(img, img_mean)

hsv = cv2.cvtColor(img_mean, cv2.COLOR_BGR2HSV)

print(hsv[:,:,0])

for y in range(hsv.shape[0]):
    for x in range(hsv.shape[1]):
        hsv_mod[y,x,0] = hsv[y, x, 0]
        hsv_mod[y,x,1] = 0
        hsv_mod[y,x,2] = 1

mod_bgr = cv2.cvtColor(hsv_mod, cv2.COLOR_HSV2BGR)

#print(mod_bgr)


cv2.imshow("Original", img)
cv2.imshow("Mean", img_mean)
cv2.imshow("HSV", hsv)
cv2.imshow("HSV_mod", hsv_mod)



cv2.waitKey(0)
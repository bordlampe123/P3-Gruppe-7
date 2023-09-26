import cv2
import numpy as np
import statistics as st

img = cv2.imread("Pierres Mappe/Kingdomino/Billeder/5.jpg")




img_out = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
img_outB = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
img_outG = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
img_outR = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
img_out2 = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
img_out3 = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
img_out4 = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
img_out5 = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)

img_hue = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

print(img_hue.shape)

def Meanifier(A, B):

    meadfind = [0]

    for y in range(5):
        starty = y*100
        print("y" + str(starty))
        for x in range(5):
            startx = x*100
            print("x" + str(startx))
            for yy in range(100):
                for xx in range(100):
                    #print(starty+yy, startx+xx)
                    meadfind.append(A[starty+yy, startx+xx, 0])
                    #print(meadfind)
            median = st.median(meadfind)
            print(median)
                    



def MeanFilterGray(A, B, z):

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

def MeanFilterColor(A, B, z):

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

#MeanFilterColor(img, img_out, 10)

Meanifier(img_hue, img_out)

cv2.imshow("Original", img)
cv2.imshow("Smoothed", img_out)

cv2.waitKey(0)
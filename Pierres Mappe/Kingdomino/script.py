import cv2
import numpy as np
import statistics as st

img = cv2.imread("Pierres Mappe/Kingdomino/Billeder/5.jpg")




img_out = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
img_outout = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

img_outB = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
img_outG = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
img_outR = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
img_out2 = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
img_out3 = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
img_out4 = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
img_out5 = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)

img_hue = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

""" def Meanifier(A, B):

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
            #print(median)
            for yy in range(100):
                for xx in range(100):
                    B[starty+yy, startx+xx, 0] = median
                    B[starty+yy, startx+xx, 1] = 150
                    B[starty+yy, startx+xx, 2] = 50
            meadfind = [0] """

def Medaifier(A, B):

    meadfindB = [0]
    meadfindG = [0]
    meadfindR = [0]

    for y in range(5):
        starty = y*100
        print("y" + str(starty))
        for x in range(5):
            startx = x*100
            print("x" + str(startx))
            for yy in range(100):
                for xx in range(100):
                    #print(starty+yy, startx+xx)
                    meadfindB.append(A[starty+yy, startx+xx, 0])
                    meadfindG.append(A[starty+yy, startx+xx, 1])
                    meadfindR.append(A[starty+yy, startx+xx, 2])
                    #print(meadfind)
            medianB = st.median(meadfindB)
            print(medianB)
            medianG = st.median(meadfindG)
            medianR = st.median(meadfindR)
            #print(median)
            for yy in range(100):
                for xx in range(100):
                    B[starty+yy, startx+xx, 0] = medianB
                    B[starty+yy, startx+xx, 1] = medianG
                    B[starty+yy, startx+xx, 2] = medianR
            meadfindB = [0]
            meadfindG = [0]
            meadfindR = [0]

def Meanifier(A, B):

    meadfindB = [0]
    meadfindG = [0]
    meadfindR = [0]

    for y in range(5):
        starty = y*100
        print("y" + str(starty))
        for x in range(5):
            startx = x*100
            print("x" + str(startx))
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

def Split(A, B, G, R):
    for y in range(A.shape[0]):
        for x in range(A.shape[1]):
            B[y, x] = A[y, x, 0]
            G[y, x] = A[y, x, 1]
            R[y, x] = A[y, x, 2]


                    



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

Meanifier(img, img_out)
Medaifier(img, img_outout)

cv2.line(img,(100,0),(100,500),(255,255,255),3)
cv2.line(img,(200,0),(200,500),(255,255,255),3)
cv2.line(img,(300,0),(300,500),(255,255,255),3)
cv2.line(img,(400,0),(400,500),(255,255,255),3)

cv2.line(img,(0,100),(500,100),(255,255,255),3)
cv2.line(img,(0,200),(500,200),(255,255,255),3)
cv2.line(img,(0,300),(500,300),(255,255,255),3)
cv2.line(img,(0,400),(500,400),(255,255,255),3)


print(img[50, 150])
print(img_out[50, 150])

Split(img_out, img_outB, img_outG, img_outR)

img_outgray = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)

img_out2 = cv2.subtract(img_outG, img_outB)
img_out3 = cv2.subtract(img_out2, img_outR)



cv2.imshow("Original", img)
cv2.imshow("Smoothed", img_out)
cv2.imshow("Blue", img_outB)
cv2.imshow("Green", img_outG)
cv2.imshow("Red", img_outR)


cv2.waitKey(0)
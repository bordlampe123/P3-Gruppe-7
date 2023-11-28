import cv2 as cv
import numpy as np

# Load the image
image = cv.imread("C:/Users/minik/Desktop/VSCode/rocks.jpg")
image = cv.resize(image, (0, 0), fx=0.2, fy=0.2)
cv.imshow("Image", image)

HSVImage = cv.cvtColor(image, cv.COLOR_BGR2HSV)
H, S, V = cv.split(HSVImage)

GrayImg = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#cv.imshow("GrayImg", GrayImg)

ContrastImg = cv.equalizeHist(GrayImg)
#cv.imshow("ContrastImg", ContrastImg)

Gaussian = cv.GaussianBlur(ContrastImg, (9, 9), 0)
cv.imshow("Gaussian", Gaussian)

Contrast2 = cv.equalizeHist(Gaussian)
cv.imshow("Contrast2", Contrast2)

Contrast2Inv = cv.bitwise_not(Contrast2)
cv.imshow("Contrast2Inv", Contrast2Inv)

#sobelx = cv.Sobel(Contrast2, cv.CV_64F, 1, 0, ksize=3)
#cv.imshow("Sobelx", sobelx)

#sobely = cv.Sobel(Contrast2, cv.CV_64F, 0, 1, ksize=3)
#cv.imshow("Sobely", sobely)

Threshold = cv.threshold(Contrast2Inv, 210, 255, cv.THRESH_BINARY)[1]
cv.imshow("Threshold", Threshold)

#Cannyedge
Canny = cv.Canny(Contrast2, 210, 220)
#cv.imshow("Canny", Canny)

#cv.imshow("H", H)
#cv.imshow("S", S)
#cv.imshow("V", V)

VHSUB = cv.subtract(V, H)
#cv.imshow("VHSUB", VHSUB)

VHH = cv.subtract(H, VHSUB)
#cv.imshow("VHH", VHH)

#thresholding
binary = cv.threshold(VHH, 80, 255, cv.THRESH_BINARY)[1]

contours = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

#drawCont = cv.drawContours(image, contours, -1, (0, 255, 0), 2)

#cv.imshow("Binary", binary)

#cv.imshow("Contours", drawCont)

min_area = 2100
unique_contours = [cnt for cnt in contours if cv.contourArea(cnt) > min_area]

print(len(unique_contours))

drawCont2 = cv.drawContours(image, unique_contours, -1, (0, 255, 0), 2)

cv.imshow("Contours2", drawCont2)

#bounding boxes
bounding_boxes = [cv.boundingRect(cnt) for cnt in unique_contours]

loc = []

for pt in bounding_boxes:
    x, y, w, h = pt
    Bounding = cv.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)
    circles = cv.circle(image, (x+w//2, y+h//2), 2, (0, 0, 255), 2)
    loc.append((x+w//2, y+h//2))

print(loc)

cv.imshow("Bounding boxes", image)
#cv.imshow("HSVImage", HSVImage)
cv.waitKey(0)
import cv2 as cv
import numpy as np

# Load the image
image = cv.imread("C:/Users/minik/Desktop/VSCode/rocks.jpg")
image = cv.resize(image, (0, 0), fx=0.2, fy=0.2)
cv.imshow("Image", image)

HSVImage = cv.cvtColor(image, cv.COLOR_BGR2HSV)
H, S, V = cv.split(HSVImage)

#cv.imshow("H", H)
#cv.imshow("S", S)
#cv.imshow("V", V)

VHSUB = cv.subtract(V, H)
#cv.imshow("VHSUB", VHSUB)

VHH = cv.subtract(H, VHSUB)
cv.imshow("VHH", VHH)

#thresholding
binary = cv.threshold(VHH, 80, 255, cv.THRESH_BINARY)[1]

contours = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

#drawCont = cv.drawContours(image, contours, -1, (0, 255, 0), 2)

cv.imshow("Binary", binary)

#cv.imshow("Contours", drawCont)



min_area = 2100
unique_contours = [cnt for cnt in contours if cv.contourArea(cnt) > min_area]

print(len(unique_contours))

drawCont2 = cv.drawContours(image, unique_contours, -1, (0, 255, 0), 2)

cv.imshow("Contours2", drawCont2)

#bounding boxes
bounding_boxes = [cv.boundingRect(cnt) for cnt in unique_contours]

for pt in bounding_boxes:
    x, y, w, h = pt
    cv.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)
    

cv.imshow("Bounding boxes", image)

print(bounding_boxes)


#cv.imshow("HSVImage", HSVImage)
cv.waitKey(0)
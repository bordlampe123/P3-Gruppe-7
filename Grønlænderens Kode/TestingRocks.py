import cv2 as cv
import numpy as np

# Load the image
image = cv.imread("Billeder/RGB_UP/Image11.jpg")
cv.imshow("Image", image)

# Convert to grayscale
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow("Gray", gray)

sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=5)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=5)
cv.imshow("Sobelx", sobelx)
cv.imshow("Sobely", sobely)

HSVImage = cv.cvtColor(image, cv.COLOR_BGR2HSV)
H, S, V = cv.split(HSVImage)

#cv.imshow("H", H)
cv.imshow("S", S)
#cv.imshow("V", V)

thresholded = cv.threshold(S, 2, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
cv.imshow("Thresholded", thresholded)


dialated = cv.dilate(thresholded, (3, 3), iterations=1)
cv.imshow("Dialated", dialated)

contours = cv.findContours(dialated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

#drawCont = cv.drawContours(image, contours, -1, (0, 255, 0), 2)

#cv.imshow("Binary", binary)
#cv.imshow("Contours", drawCont)

#printing area of contours
for cnt in contours:
    area = cv.contourArea(cnt)

small_contours = [cnt for cnt in contours if cv.contourArea(cnt) < 200]

mask = np.zeros_like(thresholded)
cv.drawContours(mask, small_contours, -1, (255, 255, 255), -1)
cv.imshow("Mask", mask)

subtractedSmall = cv.subtract(thresholded, mask)
cv.imshow("Subtracted", subtractedSmall)

dialated2 = cv.dilate(subtractedSmall, (3, 3), iterations=6)
cv.imshow("Dialated2", dialated2)

contours2 = cv.findContours(dialated2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
#drawCont2 = cv.drawContours(image, contours2, -1, (0, 255, 0), 2)

#cv.imshow("Contours2", drawCont2)
ellipse_img = np.zeros_like(image)
roi = []

for cnt in contours2:
    contour_area = cv.contourArea(cnt)

    ellipse = cv.fitEllipse(cnt)
    #for ellipse in [ellipse]:
        #print(ellipse)
    #cv.ellipse(image, ellipse, (0, 255, 0), 2)

    x, y, w, h = cv.boundingRect(cnt)
    #cv.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)
    roi.append(image[y:y+h, x:x+w])
      
for img in roi:
    meanshift = cv.pyrMeanShiftFiltering(img, 2, 40)
    gauss = cv.GaussianBlur(meanshift, (3, 3), 0)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    equalis = cv.equalizeHist(gray)
    cv.imshow("ROI", equalis)
    cv.imshow("ROI2", meanshift)
    cv.imshow("ROI3", gauss)
    cv.waitKey(1)



cv.imshow("Image", image)


#min_area = 2100
#unique_contours = [cnt for cnt in contours if cv.contourArea(cnt) > min_area]

#print(len(unique_contours))

#drawCont2 = cv.drawContours(image, unique_contours, -1, (0, 255, 0), 2)

#cv.imshow("Contours2", drawCont2)

#bounding boxes
#bounding_boxes = [cv.boundingRect(cnt) for cnt in unique_contours]

loc = []

#for pt in bounding_boxes:
    #x, y, w, h = pt
    #Bounding = cv.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)
    #circles = cv.circle(image, (x+w//2, y+h//2), 2, (0, 0, 255), 2)
    #loc.append((x+w//2, y+h//2))

#print(loc)

#cv.imshow("Bounding boxes", image)
#cv.imshow("HSVImage", HSVImage)
cv.waitKey(0)
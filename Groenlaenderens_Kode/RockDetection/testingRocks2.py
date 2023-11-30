import cv2 as cv
import numpy as np

image = cv.imread("C:/Users/minik/Desktop/VSCode/GIt/P3-Gruppe-7/Groenlaenderens_Kode/RockDetection/Billeder/Image6.jpg")
image2 = image.copy()
img3 = image.copy()

image5 = cv.cvtColor(image, cv.COLOR_BGR2Luv)

img_h, img_w = image.shape[:2]
cv.imshow("Image", image)
cv.imshow("Image5", image5)

a, b, c = cv.split(image5)

cv.imshow("c", c)

contrastC = cv.equalizeHist(c)
C = cv.merge((contrastC, contrastC, contrastC))
print(C.shape)
meanshiftc = cv.pyrMeanShiftFiltering(C, 10, 20)
cv.imshow("meanshiftc", meanshiftc)

invmeanshiftc = cv.bitwise_not(meanshiftc)
cv.imshow("invmeanshiftc", invmeanshiftc)
invmeanshiftc = np.uint8(invmeanshiftc)

normalized = cv.normalize(invmeanshiftc, None, 0, 255, cv.NORM_MINMAX)

Gray = cv.cvtColor(normalized, cv.COLOR_BGR2GRAY)
cv.imshow("Gray", Gray)
print(Gray.shape)
print(Gray.dtype)
# Apply thresholding
ret, Cthresh = cv.threshold(Gray, 223, 255, cv.THRESH_BINARY)

# Convert to uint8
Cthresh = np.uint8(Cthresh)

mask = np.zeros_like(Cthresh)
contours = cv.findContours(Cthresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

for cnt in contours:
    area = cv.contourArea(cnt)
    if area > 500:
        cv.drawContours(mask, [cnt], -1, (255, 255, 255), -1)

cv.imshow("Mask", mask)

cv.imshow("Cthresh", Cthresh)
cv.waitKey(0)
cv.destroyAllWindows()

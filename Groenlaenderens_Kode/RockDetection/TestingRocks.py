import cv2 as cv
import numpy as np

# Load the image
image = cv.imread("C:/Users/minik/Desktop/VSCode/GIt/P3-Gruppe-7/Groenlaenderens_Kode/RockDetection/Billeder/Image3.jpg")

image2 = image.copy()

cv.imshow("Image", image)

# Convert to grayscale
gray = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=5)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=5)

normSobelx = cv.normalize(sobelx, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
normSobely = cv.normalize(sobely, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

SquaredSobelx = cv.multiply(sobelx, sobelx)
SquaredSobely = cv.multiply(sobely, sobely)

SqrtSobel = cv.sqrt(SquaredSobelx + SquaredSobely + SquaredSobelx + SquaredSobely)
Sobel = cv.add(normSobelx, normSobely)
NormSqrtSob = cv.normalize(SqrtSobel, None, 0, 170, cv.NORM_MINMAX, cv.CV_8U)

cv.imshow("SqrtSobelx", SqrtSobel)
cv.imshow("Sobel", Sobel)
cv.imshow("NormSqrtSob", NormSqrtSob)

cv.imshow("SquaredSobelx", SquaredSobelx)
cv.imshow("SquaredSobely", SquaredSobely)

cv.imshow("Sobelx2", normSobelx)
cv.imshow("Sobely2", normSobely)

cv.imshow("Sobelx", sobelx)
cv.imshow("Sobely", sobely)
cv.waitKey(0)

HSVImage = cv.cvtColor(image2, cv.COLOR_BGR2HSV)
H, S, V = cv.split(HSVImage)

thresholded = cv.threshold(S, 2, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

dilated = cv.dilate(thresholded, (3, 3), iterations=1)

contours = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

# printing area of contours
small_contours = [cnt for cnt in contours if cv.contourArea(cnt) < 200]

mask = np.zeros_like(thresholded)
cv.drawContours(mask, small_contours, -1, (255, 255, 255), -1)

subtractedSmall = cv.subtract(thresholded, mask)

dilated2 = cv.dilate(subtractedSmall, (3, 3), iterations=6)

contours2 = cv.findContours(dilated2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

ellipse_img = np.zeros_like(image)
roi = []
ellipsis = []

for cnt in contours2:
    contour_area = cv.contourArea(cnt)
    ellipse = cv.fitEllipse(cnt)
    ellipsis.append(ellipse)
    cv.ellipse(image2, ellipse, (0, 255, 0), 2)

    x, y, w, h = cv.boundingRect(cnt)
    Box = cv.boxPoints(ellipse)
    print(Box)
    cv.rectangle(image2, (x-10, y-10), (x+w+10, y+h+10), (0, 0, 255), 2)
    roi.append(image[y-10:y+h+10, x-10:x+w+10])

for img in roi:
    cv.destroyAllWindows()
    cv.imshow("ROI", img)
    cv.waitKey(0)

cv.imshow("Image", image)
cv.imshow("Image2", image2)
cv.waitKey(0)
cv.destroyAllWindows()
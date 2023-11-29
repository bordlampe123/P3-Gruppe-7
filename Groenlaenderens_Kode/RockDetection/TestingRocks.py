import cv2 as cv
import numpy as np
import skimage.exposure as exposure

# Load the image
image = cv.imread("C:/Users/minik/Desktop/VSCode/GIt/P3-Gruppe-7/Groenlaenderens_Kode/RockDetection/Billeder/Image3.jpg")

image2 = image.copy()

cv.imshow("Image", image)

# Convert to grayscale
gray = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

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
i = 1
for cnt in contours2:
    ellipse = cv.fitEllipse(cnt)
    
    # Convert ellipse coordinates to integers
    x_coord = int(ellipse[0][0])
    y_coord = int(ellipse[0][1])

    cv.putText(image2, str(i), (x_coord, y_coord), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    ellipsis.append(ellipse)
    cv.ellipse(image2, ellipse, (0, 255, 0), 2)
    i += 1

    x, y, w, h = cv.boundingRect(cnt)
    cv.rectangle(image2, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi.append(image[y-10:y+h+10, x-10:x+w+10])

for img in roi:
    cv.destroyAllWindows()
    cv.imshow("ROI", img)
    cv.waitKey(0)

print("Number of rocks: ", len(roi))

cv.imshow("Image", image)
cv.imshow("Image2", image2)
cv.waitKey(0)
cv.destroyAllWindows()
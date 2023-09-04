#Opencv Test



import cv2 as cv
img = cv.imread("C:/Users/bebj2/OneDrive/Skrivebord/Ordnede billeder/Bund2.png", 1)
img = cv.resize(img, (0,0), fx=1/5, fy=1/5)
img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)

cv.imwrite("NyVersion.png", img)

cv.imshow("Display window", img)
k = cv.waitKey(0) # Wait for a keystroke in the window


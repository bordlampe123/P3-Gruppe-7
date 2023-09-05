#Opencv Test


import numpy as np
import cv2 as cv
#img = cv.imread("C:/Users/bebj2/OneDrive/Skrivebord/Ordnede billeder/Bund1.png", -1)
img = cv.imread("C:/Users/bebj2/Downloads/tinypic.png", -1)
assert img is not None, "file could not be read, check with os.path.exists()"
[x,y] = img.shape
#img = cv.resize(img, (0,0), fx=1/5, fy=1/5)
#img = cv.resize(img, (200,200))
#img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)

#cv.imwrite("NyVersion.png", img)





print(img)
print("cunt")
#cv.imshow("Display window", img)
k = cv.waitKey(0) # Wait for a keystroke in the window


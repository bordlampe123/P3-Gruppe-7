import cv2 as cv
import numpy as np
import math

# Load the image
image = cv.imread("Vores_MiniProjekt/King Domino dataset/King Domino dataset/Cropped and perspective corrected boards/10.jpg")
cv.imshow("board", image)   

# Split the image into RGB channels
B, G, R = cv.split(image)

# Subtract the red channel from the green channel
GRSub = cv.subtract(G, R)
#cv.imshow("GRSub", GRSub)

# Subtract the green channel from the blue channel
BGSub = cv.subtract(B, G)

# Subtract the blue channel from the red channel
RBSub = cv.subtract(R, B)

# Subtract GRSub from the blue channel
BSub = cv.subtract(B, GRSub)
#cv.imshow("BSub", BSub)

# Subtracting BGSub from BSub
BGBSub = cv.subtract(BSub, BGSub)
#cv.imshow("BGBub", BGBSub)

# Subtracting RBSub from GBGSub
FinalImage = cv.subtract(BGBSub, RBSub)
#cv.imshow("FinalImage", FinalImage)

# Convert the image to the grayscale color space
gray_img = cv.merge((FinalImage, FinalImage, FinalImage))
cv.imshow("gray_img", gray_img)

# Template
template = cv.imread("C:/Users/minik/Desktop/VSCode/SubtractCrown2.jpg", cv.IMREAD_COLOR)

# Blurred template
template2 = cv.GaussianBlur(template, (7, 7), 0)

gray_img2 = cv.GaussianBlur(gray_img, (7, 7), 0)
#v.imshow("template", gray_img2)

# Create an array with the templates rotated 90, 180 and 270 degrees
Templates = [template, cv.rotate(template, cv.ROTATE_90_CLOCKWISE), cv.rotate(template, cv.ROTATE_180), cv.rotate(template, cv.ROTATE_90_COUNTERCLOCKWISE)]

Templates2 = [template2, cv.rotate(template2, cv.ROTATE_90_CLOCKWISE), cv.rotate(template2, cv.ROTATE_180), cv.rotate(template2, cv.ROTATE_90_COUNTERCLOCKWISE)]

template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

# Store width and height of image and template in w and h
w, h, channels = image.shape

print(image.shape)

tw, th = template.shape

# Creating an empty array for later scoring
scores = np.zeros((5, 5))

# Creating a list of all the subimages
subimage_height = h // 5
subimage_width = w // 5

subimages = []
for i in range(5):
    for j in range(5):
        x_start = i * subimage_height
        x_end = (i + 1) * subimage_height
        y_start = j * subimage_width
        y_end = (j + 1) * subimage_width
        subimage = gray_img2[x_start:x_end, y_start:y_end]
        subimages.append(subimage)

#Template matching through each subimage
for i in range(len(subimages)):
    for j in range(len(Templates2)):
        res = cv.matchTemplate(subimages[i], Templates2[j], cv.TM_CCOEFF_NORMED)
        loc = np.where(res >= 0.73)

        for pt in zip(*loc[::-1]):
            cv.rectangle(subimages[i], pt, (pt[0] + tw, pt[1] + th), (0, 255, 255), 2)

        
cv.imshow("image", gray_img2)

cv.waitKey(0)


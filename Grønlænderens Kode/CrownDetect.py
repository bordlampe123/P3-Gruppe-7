import cv2 as cv
import numpy as np
import math
import os

# Load the image
image = cv.imread("Vores_MiniProjekt/King Domino dataset/King Domino dataset/Cropped and perspective corrected boards/1.jpg")
cv.imshow("board", image)   

# Split the image into RGB channels
B, G, R = cv.split(image)

# Subtract the green channel from the red channel - Mortens
GBSub = cv.subtract(G, B)

# Subtract the blue channel from the red channel - Mortens
BRSub = cv.subtract(B, R)

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

BRBSub = cv.subtract(B, RBSub)

# Subtracting BRBSub from BRSub - Mortens
FinalImage2 = cv.subtract(BRBSub, BRSub)  

# Convert the image to the grayscale color space
gray_img = cv.merge((FinalImage, FinalImage, FinalImage))

cv.imshow("gray_img", gray_img)

# Template
template = cv.imread("C:/Users/minik/Desktop/VSCode/Crown_bw.png", cv.IMREAD_COLOR)

#templateL = cv.imread("C:/Users/minik/Desktop/VSCode/Crown_L.png", cv.IMREAD_COLOR)

#templateR = cv.imread("C:/Users/minik/Desktop/VSCode/Crown_R.png", cv.IMREAD_COLOR)

#templateD = cv.imread("C:/Users/minik/Desktop/VSCode/Crown_D.png", cv.IMREAD_COLOR)

# Blurred template
#template2 = cv.GaussianBlur(template, (7, 7), 0)

gray_img2 = cv.GaussianBlur(gray_img, (7, 7), 0)
#v.imshow("template", gray_img2)

# Create an array with the templates rotated 90, 180 and 270 degrees
Templates = [template, cv.rotate(template, cv.ROTATE_90_CLOCKWISE), cv.rotate(template, cv.ROTATE_180), cv.rotate(template, cv.ROTATE_90_COUNTERCLOCKWISE)]

template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
#Templates = [template, templateL, templateD, templateR]

#Templates2 = [template2, cv.rotate(template2, cv.ROTATE_90_CLOCKWISE), cv.rotate(template2, cv.ROTATE_180), cv.rotate(template2, cv.ROTATE_90_COUNTERCLOCKWISE)]

# Store width and height of image and template in w and h
w, h, channels = image.shape

print(image.shape)

tw, th = template.shape

# Creating a list of all the subimages
subimage_height = h // 5
subimage_width = w // 5

subimages = np.zeros((5,5), dtype=object)
for i in range(5):
    for j in range(5):
        x_start = i * subimage_height
        x_end = (i + 1) * subimage_height
        y_start = j * subimage_width
        y_end = (j + 1) * subimage_width
        subimage = gray_img[x_start:x_end, y_start:y_end]
        
        subimages[i,j] = subimage
        # Store the subimage in an array

# Creating an array to store the number of matches for each template in each subimage       
matches = np.zeros((5,5), dtype=object)
distance_threshold = 10

#Template matching through each subimage
for i in range(5):
    for j in range(5):

        SubimageMatches = [] # List of matches for each subimage

        MatchCount = 0 # Number of matches for each template in each subimage

        for k in range(len(Templates)):
            res = cv.matchTemplate(subimages[i,j], Templates[k], cv.TM_CCOEFF_NORMED)
            loc = np.where(res >= 0.57)
            
            MatchPoints = list(zip(*loc[::-1])) # List of points where the template matches the subimage

            UniquePoints = [] # List of unique points

            for match in MatchPoints:
                unique = True
                for entries in UniquePoints:
                    if math.sqrt((match[0] - entries[0])**2 + (match[1] - entries[1])**2) < distance_threshold:
                        unique = False
                        break
                if unique:
                    UniquePoints.append(match)

            MatchCount += len(UniquePoints) # Number of matches for each template in each subimage

            for pt in zip(*loc[::-1]):
                cv.rectangle(subimages[i,j], pt, (pt[0] + tw, pt[1] + th), (0, 255, 255), 2)

        matches[i,j] = MatchCount


print(matches)

# Sorting matches and storing them in a list






cv.imshow("image", gray_img)

cv.waitKey(0)


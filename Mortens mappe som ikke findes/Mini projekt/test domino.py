# Python program to illustrate
# template matching
import cv2
import numpy as np

# Read the main image
img_rgb = cv2.imread('C:/Users/stron/PycharmProjects/P3-Gruppe-7/Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Result of 1.jpg (blue).png')
# Convert it to grayscale
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)


# placing marks where template 2 was found
# Read the template
template2 = cv2.imread(
	'/Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Crown templates/Crown facing left.png', 0)

# Store width and height of template in w and h
w2, h2 = template2.shape[::-1]

# Perform match operations.
res = cv2.matchTemplate(img_gray, template2, cv2.TM_CCOEFF_NORMED)

# Specify a threshold
threshold = 0.7

# Store the coordinates of matched area in a numpy array
loc2 = np.where(res >= threshold)


# for template 2
# Draw a rectangle around the matched region.
for pt in zip(*loc2[::-1]):
	cv2.rectangle(img_rgb, pt, (pt[0] + w2, pt[1] + h2), (0, 255, 255), 2)



# Show the final image with the matched area.
cv2.imshow('Detected', img_rgb)
cv2.waitKey(0)

import cv2
import numpy as np

# Read the main image

#img = cv2.imread("C:/Users/stron/PycharmProjects/P3-Gruppe-7/Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Cropped and perspective corrected boards/1.jpg")
#img = cv2.imread("C:/Users/stron/PycharmProjects/P3-Gruppe-7/Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Cropped and perspective corrected boards/2.jpg")
#img = cv2.imread("C:/Users/stron/PycharmProjects/P3-Gruppe-7/Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Cropped and perspective corrected boards/3.jpg")
#img = cv2.imread("C:/Users/stron/PycharmProjects/P3-Gruppe-7/Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Cropped and perspective corrected boards/4.jpg")
#img = cv2.imread("C:/Users/stron/PycharmProjects/P3-Gruppe-7/Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Cropped and perspective corrected boards/5.jpg")
#img = cv2.imread("C:/Users/stron/PycharmProjects/P3-Gruppe-7/Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Cropped and perspective corrected boards/6.jpg")
#img = cv2.imread("C:/Users/stron/PycharmProjects/P3-Gruppe-7/Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Cropped and perspective corrected boards/7.jpg")
#img = cv2.imread("C:/Users/stron/PycharmProjects/P3-Gruppe-7/Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Cropped and perspective corrected boards/8.jpg")
#img = cv2.imread("C:/Users/stron/PycharmProjects/P3-Gruppe-7/Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Cropped and perspective corrected boards/9.jpg")
#img = cv2.imread("C:/Users/stron/PycharmProjects/P3-Gruppe-7/Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Cropped and perspective corrected boards/10.jpg")
#img = cv2.imread("C:/Users/stron/PycharmProjects/P3-Gruppe-7/Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Cropped and perspective corrected boards/10.jpg")
#img = cv2.imread("C:/Users/stron/PycharmProjects/P3-Gruppe-7/Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Cropped and perspective corrected boards/11.jpg")
#img = cv2.imread("C:/Users/stron/PycharmProjects/P3-Gruppe-7/Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Cropped and perspective corrected boards/12.jpg")
#img = cv2.imread("C:/Users/stron/PycharmProjects/P3-Gruppe-7/Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Cropped and perspective corrected boards/13.jpg")
#img = cv2.imread("C:/Users/stron/PycharmProjects/P3-Gruppe-7/Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Cropped and perspective corrected boards/14.jpg")
#img = cv2.imread("C:/Users/stron/PycharmProjects/P3-Gruppe-7/Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Cropped and perspective corrected boards/15.jpg")
#img = cv2.imread("C:/Users/stron/PycharmProjects/P3-Gruppe-7/Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Cropped and perspective corrected boards/16.jpg")
#img = cv2.imread("C:/Users/stron/PycharmProjects/P3-Gruppe-7/Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Cropped and perspective corrected boards/17.jpg")
#img = cv2.imread("C:/Users/stron/PycharmProjects/P3-Gruppe-7/Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Cropped and perspective corrected boards/18.jpg")
#img = cv2.imread("C:/Users/stron/PycharmProjects/P3-Gruppe-7/Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Cropped and perspective corrected boards/19.jpg")
#img = cv2.imread("C:/Users/stron/PycharmProjects/P3-Gruppe-7/Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Cropped and perspective corrected boards/20.jpg")
#img = cv2.imread("C:/Users/stron/PycharmProjects/P3-Gruppe-7/Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Cropped and perspective corrected boards/21.jpg")
#img = cv2.imread("C:/Users/stron/PycharmProjects/P3-Gruppe-7/Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Cropped and perspective corrected boards/22.jpg")
#img = cv2.imread("C:/Users/stron/PycharmProjects/P3-Gruppe-7/Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Cropped and perspective corrected boards/23.jpg")
#img = cv2.imread("C:/Users/stron/PycharmProjects/P3-Gruppe-7/Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Cropped and perspective corrected boards/24.jpg")
#img = cv2.imread("C:/Users/stron/PycharmProjects/P3-Gruppe-7/Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Cropped and perspective corrected boards/25.jpg")
#img = cv2.imread("C:/Users/stron/PycharmProjects/P3-Gruppe-7/Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Cropped and perspective corrected boards/26.jpg")
#img = cv2.imread("C:/Users/stron/PycharmProjects/P3-Gruppe-7/Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Cropped and perspective corrected boards/27.jpg")
#img = cv2.imread("C:/Users/stron/PycharmProjects/P3-Gruppe-7/Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Cropped and perspective corrected boards/28.jpg")
#img = cv2.imread("C:/Users/stron/PycharmProjects/P3-Gruppe-7/Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Cropped and perspective corrected boards/29.jpg")
#img = cv2.imread("C:/Users/stron/PycharmProjects/P3-Gruppe-7/Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Cropped and perspective corrected boards/30.jpg")
#img = cv2.imread("C:/Users/stron/PycharmProjects/P3-Gruppe-7/Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Cropped and perspective corrected boards/31.jpg")


#make new images to place the new color chanels into

img_outB = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
img_outG = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
img_outR = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
img_out2 = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
img_out3 = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
img_out4 = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
img_out5 = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)

#a function that splits the original image into its 3 color chanels
def Split(A, B, G, R):
    for y in range(A.shape[0]):
        for x in range(A.shape[1]):
            B[y, x] = A[y, x, 0]
            G[y, x] = A[y, x, 1]
            R[y, x] = A[y, x, 2]


#the original image it split
Split(img, img_outB, img_outG, img_outR)

#the diferent color chanels are subtracted from one another in order to end up ith only the white colors form the original image
img_out2 = cv2.subtract(img_outG, img_outB)

img_out3 = cv2.subtract(img_outR, img_outB)

img_out4 = cv2.subtract(img_outB, img_outR)

img_out5 = cv2.subtract(img_outB, img_out3)

img_rgb = cv2.subtract(img_out5, img_out4)

#img_out6 = cv2.subtract(img_outR, img_outB)

#img_rgb = cv2.subtract(img_out3, img_out4)


#bugfix showing the created images only for debugging
#cv2.imshow("Only Blue", img_outB)
#cv2.imshow("Only Green", img_outG)
#cv2.imshow("Only Red", img_outR)
cv2.imshow("Original", img)
#cv2.waitKey(0)



Crown1 = "C:/Users/stron/PycharmProjects/P3-Gruppe-7/Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Crown templates/Crown_bw.png"
#placing marks where template 1 was found
# Read the template
template1 = cv2.imread(Crown1, 0)

# Store width and height of template in w and h
w1, h1 = template1.shape[::-1]

# Perform match operations.
res = cv2.matchTemplate(img_rgb, template1, cv2.TM_CCOEFF_NORMED)

# Specify a threshold
threshold = 0.6

# Store the coordinates of matched area in a numpy array
loc1 = np.where(res >= threshold)


Crown2 = "C:/Users/stron/PycharmProjects/P3-Gruppe-7/Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Crown templates/Crown facing left.png"
# placing marks where template 2 was found
# Read the template
template2 = cv2.imread(Crown2, 0)

# Store width and height of template in w and h
w2, h2 = template2.shape[::-1]

# Perform match operations.
res = cv2.matchTemplate(img_rgb, template2, cv2.TM_CCOEFF_NORMED)

# Specify a threshold
threshold = 0.6

# Store the coordinates of matched area in a numpy array
loc2 = np.where(res >= threshold)



Crown3 = "C:/Users/stron/PycharmProjects/P3-Gruppe-7/Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Crown templates/Crown_facing_right.png"
# placing marks where template 3 was found
# Read the template
template3 = cv2.imread(Crown3, 0)

# Store width and height of template in w and h
w3, h3 = template3.shape[::-1]

# Perform match operations.
res = cv2.matchTemplate(img_rgb, template3, cv2.TM_CCOEFF_NORMED)

# Specify a threshold
threshold = 0.6

# Store the coordinates of matched area in a numpy array
loc3 = np.where(res >= threshold)




Crown4 = "C:/Users/stron/PycharmProjects/P3-Gruppe-7/Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Crown templates/Crown_facing_down.png"
# placing marks where template 4 was found
# Read the template
template4 = cv2.imread(Crown4, 0)

# Store width and height of template in w and h
w4, h4 = template4.shape[::-1]

# Perform match operations.
res = cv2.matchTemplate(img_rgb, template4, cv2.TM_CCOEFF_NORMED)

# Specify a threshold
threshold = 0.6

# Store the coordinates of matched area in a numpy array
loc4 = np.where(res >= threshold)





#now boxes are drawn around the crowns

# for template 1
# Draw a rectangle around the matched region.
for pt in zip(*loc1[::-1]):
	cv2.rectangle(img_rgb, pt, (pt[0] + w1, pt[1] + h1), (255, 50, 255), 2)




# for template 2
# Draw a rectangle around the matched region.
for pt in zip(*loc2[::-1]):
	cv2.rectangle(img_rgb, pt, (pt[0] + w2, pt[1] + h2), (255, 50, 255), 2)




# for template 3
# Draw a rectangle around the matched region.
for pt in zip(*loc3[::-1]):
	cv2.rectangle(img_rgb, pt, (pt[0] + w3, pt[1] + h3), (255, 50, 255), 2)



# for template 4
# Draw a rectangle around the matched region.
for pt in zip(*loc4[::-1]):
	cv2.rectangle(img_rgb, pt, (pt[0] + w4, pt[1] + h4), (255, 50, 255), 2)
cv2.rectangle



# Show the final image with the matched area.
cv2.imshow('Detected', img_rgb)
cv2.waitKey(0)
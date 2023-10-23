import cv2
import numpy as np

#test ideer:
    #forsøg at sharpen images inden de bliver opdelt,
    #se på right template for at få image 46 til at virke
    #leg evt med tresholds

# Read the main image
img = cv2.imread("C:/Users/stron/PycharmProjects/P3-Gruppe-7/Mortens mappe som ikke findes/Mini projekt/King Domino dataset/Cropped and perspective corrected boards/48.jpg")

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


# Store the coordinates of matched area in a numpy array
loc4 = np.where(res >= threshold)


# a nice function taht filters multiple detections of the same crown out so we only detect each one once
def filter_close_points(points, threshold):
    result = []
    for i in range(len(points)):
        x1, y1 = points[i]
        keep = True
        for j in range(i + 1, len(points)):
            x2, y2 = points[j]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if distance < threshold:
                keep = False
                break
        if keep:
            result.append((x1, y1))
    return result


#here the crowns are filterd using the filter_close_points fuction that looks at the next point in the array and determines if it is the same crow
result1 = filter_close_points(np.column_stack(loc1), 5)

result2 = filter_close_points(np.column_stack(loc2), 5)

result3 = filter_close_points(np.column_stack(loc3), 5)

result4 = filter_close_points(np.column_stack(loc4), 5)



all_results = result1 + result2 + result3 + result4
all_results.sort()



def filter_close_pointstupe(points, threshold):
    result = []
    for i in range(len(points)):
        x1, y1 = points[i]
        keep = True
        for j in range(i + 1, len(points)):
            x2, y2 = points[j]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if distance < threshold:
                keep = False
                break
        if keep:
            result.append((x1, y1))
    return result
Final_Results = filter_close_pointstupe(all_results, 5)

print(Final_Results)



#now boxes are drawn around the crowns
 #for template 1
 #Draw a rectangle around the matched region.
for y, x in result1:
    cv2.rectangle(img, (x, y), (x + w1, y + h1), (255, 50, 255), 2)


for y, x in result2:
    cv2.rectangle(img, (x, y), (x + w2, y + h2), (255, 50, 255), 2)

for y, x in result3:
    cv2.rectangle(img, (x, y), (x + w3, y + h3), (255, 50, 255), 2)

for y, x in result4:
    cv2.rectangle(img, (x, y), (x + w4, y + h4), (255, 50, 255), 2)


# Show the final image with the matched area.
cv2.imshow('Detected', img)

cv2.waitKey(0)

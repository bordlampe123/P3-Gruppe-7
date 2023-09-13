import cv2

# Open picture
img = cv2.imread("C:/Users/pierr/Desktop/P3-cunts/Pierres shit/Computer Vision/tinypic.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Our window", img)

# Define height and width in pixel
height = img.shape[0]
width = img.shape[1]

print("Højde på billede: " + str(height) + ", Bredde på billede: " + str(width))

# Go through all the x pixels
for i in range(width):
    for j in range(height):
        k = img[i, j]
        print(k)

# Display the picture
cv2.imshow("Our window", img)
cv2.waitKey(0)

print("Hjælp")
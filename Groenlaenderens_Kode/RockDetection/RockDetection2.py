import cv2 as cv
import numpy as np
import skimage.exposure as exposure

#Load image and get image dimensions
image = cv.imread("C:/Users/minik/Desktop/VSCode/GIt/P3-Gruppe-7/Groenlaenderens_Kode/RockDetection/Billeder/Image11.jpg")
image2 = image.copy()
img3 = image.copy()
img_h, img_w = image.shape[:2]

# Convert to HSV and split into channels
HSVImage = cv.cvtColor(image2, cv.COLOR_BGR2HSV)
H, S, V = cv.split(HSVImage)
cv.imshow("S", S)   

# Thresholding the saturation channel
thresholded = cv.threshold(S, 31.9, 255, cv.THRESH_BINARY)[1]
#cv.imshow("Thresholded", thresholded)

# Dilate thresholded image and find contours
dilated = cv.dilate(thresholded, (3, 3), iterations=3)
cv.imshow("Dilated", dilated)
contours = cv.findContours(thresholded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

#Finding small contours and drawing them on a mask
small_contours = [cnt for cnt in contours if cv.contourArea(cnt) < 1000]
mask = np.zeros_like(thresholded)
cv.drawContours(mask, small_contours, -1, (255, 255, 255), -1)
#cv.imshow("Mask", mask)
#cv.waitKey(0)

#Subtracting the small contours from the thresholded image
subtractedSmall = cv.subtract(thresholded, mask)
dialated2 = cv.dilate(subtractedSmall, (3, 3), iterations=2)
contours2 = cv.findContours(dialated2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
contour_img = np.zeros_like(image)

#Drawing filled contours on the contour_img
drawCont = cv.drawContours(contour_img, contours2, -1, (255, 255, 255), -1)
cv.imshow("Contours", contour_img)

#Lists for sorting contours and ROI
InBoundRock = []
InBoundCont = []
OutBoundRock = []
SortHull = []
SolidityThres = 0.88
EllipseThres = 0.80
MultipleRocks = []


for cnt in contours2:
    Area = cv.contourArea(cnt)
    SortingConvexHull = cv.convexHull(cnt)
    SortingEllipse = cv.fitEllipse(SortingConvexHull)
    ConvexHullArea = cv.contourArea(SortingConvexHull)
    SortingEllipseArea = (SortingEllipse[1][0]/2)*(SortingEllipse[1][1]/2)*np.pi
    Solidity = float(Area)/ConvexHullArea
    SolidityEllipse = float(Area)/SortingEllipseArea
    print("Solidity" + " " + str(Solidity))
    print("Ellipse" + " " + str(SolidityEllipse))
    cv.ellipse(image2, SortingEllipse, (0, 255, 0), 2)
    cv.imshow("Image2", image2)
    cv.waitKey(0)
    if 15 < SortingEllipse[0][0] < img_w-15 and 15 < SortingEllipse[0][1] < img_h-15:
        InBoundRock.append(SortingEllipse)
        InBoundCont.append(cnt)
        SortHull.append(SortingConvexHull)
        if Solidity < SolidityThres and SolidityEllipse < EllipseThres:
            MultipleRocks.append(SortingEllipse)
    else:
        OutBoundRock.append(SortingEllipse)



cv.rectangle(image2, (15, 15), (img_w-15, img_h-15), (0, 0, 255), 2)

#print("Inbound Rock Count:", len(InBoundRock))
#print("Outbound Rock Count:", len(OutBoundRock))
cv.waitKey(0)

cv.imshow("drawCont", drawCont) 
cv.waitKey(0)
#Morphological opening
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(contour_img, cv.MORPH_OPEN, kernel, iterations=10)
opneing = np.uint8(opening)
cv.imshow("Opening", opening)
opening_gray = cv.cvtColor(opening, cv.COLOR_BGR2GRAY)

#Finding sure background and foreground through dilation and distance transform thresholding
sure_bg = cv.dilate(opening, kernel, iterations=2)
cv.imshow("Sure_bg", sure_bg)

#Distance transform, distance from each pixel to the nearest zero pixel, normalized to fit in 0-255
distance_transform = cv.distanceTransform(opening_gray, cv.DIST_L2, 5)
normalized_distance = exposure.rescale_intensity(distance_transform, out_range=(0, 255))
normalized_distance = normalized_distance.astype(np.uint8)
cv.imshow("Normalized Distance", normalized_distance)
#cv.imshow("Distance Transform", distance_transform)
#cv.waitKey(0)

_, distThres = cv.threshold(distance_transform, 17, 255, cv.THRESH_BINARY)
cv.imshow("Distance Threshold", distThres)


sure_bg = cv.dilate(opening, kernel, iterations=2)
cv.imshow("Sure_bg", sure_bg)
sure_bGray = cv.cvtColor(sure_bg, cv.COLOR_BGR2GRAY)

sure_bGray = np.uint8(sure_bGray)
distThres = np.uint8(distThres)


unknown = cv.subtract(sure_bGray, distThres)
cv.imshow("Unknown", unknown)

labels = cv.connectedComponents(distThres, connectivity=8, ltype=cv.CV_32S)[1]
labelVis = cv.applyColorMap(np.uint8(labels), cv.COLORMAP_JET)
labels = labels + 1
labels[unknown == 255] = 0

labels = cv.watershed(image2, labels)
cv.imshow("Image2", np.uint8(labels))
image2[labels == -1] = [255, 0, 0]

UnLabels = np.unique(labels)

AllCont = []

for i in range(len(UnLabels)):
    if i == 0 or i == -1 or i == 1:
        continue
    else:
        mask = np.zeros_like(image2)
        mask[labels == i] = 255
        maskThresh = cv.threshold(mask, 1, 255, cv.THRESH_BINARY)[1]
        maskThresh = np.uint8(maskThresh)
        maskThresh = cv.cvtColor(maskThresh, cv.COLOR_BGR2GRAY)
        contours, _ = cv.findContours(maskThresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        AllCont.extend(contours)
        drawCont = cv.drawContours(image2, contours, -1, (0, 255, 0), 2)
        continue

AllEllips = []
for cont in AllCont:
    hull = cv.convexHull(cont)
    ellipsis = cv.fitEllipse(hull)
    cv.ellipse(img3, ellipsis, (0, 255, 0), 2)
    AllEllips.append(ellipsis)


print("Rock Count:", len(AllEllips))


cv.imshow("labels2", np.uint8(labels))

cv.imshow("Image2", image2)
cv.imshow("Image3", img3)
cv.imshow("Labels", labelVis)
cv.waitKey(0)
cv.destroyAllWindows()
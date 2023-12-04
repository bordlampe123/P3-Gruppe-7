import cv2 as cv
import numpy as np
import skimage.exposure as exposure

image = cv.imread("Groenlaenderens_Kode/RockDetection/Billeder/Image4.jpg")
image2 = image.copy()
img3 = image.copy()

img_h, img_w = image.shape[:2]

# Convert to grayscale0
gray = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
HSVImage = cv.cvtColor(image2, cv.COLOR_BGR2HSV)
H, S, V = cv.split(HSVImage)

thresholded = cv.threshold(S, 10, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
#cv.imshow("Thresholded", thresholded)

dilated = cv.dilate(thresholded, (3, 3), iterations=1)

contours = cv.findContours(thresholded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
#cv.imshow("Dilated", thresholded)
#cv.waitKey(0)
# printing area of contours
small_contours = [cnt for cnt in contours if cv.contourArea(cnt) < 500]

mask = np.zeros_like(thresholded)
cv.drawContours(mask, small_contours, -1, (255, 255, 255), -1)
#cv.imshow("Mask", mask)
#cv.waitKey(0)

subtractedSmall = cv.subtract(thresholded, mask)

dilated2 = cv.dilate(subtractedSmall, (5, 5), iterations=2)
#cv.imshow("Dilated2", dilated2)
#cv.waitKey(0)

contour_img = np.zeros_like(image)

contours2 = cv.findContours(dilated2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
drawCont = cv.drawContours(contour_img, contours2, -1, (255, 255, 255), -1)
cv.imshow("Contours", contour_img)
print(contours2[0])
cv.waitKey(0)

kernel = np.ones((3, 3), np.uint8)
kernel2 = np.ones((11, 11), np.uint8)
opening = cv.morphologyEx(contour_img, cv.MORPH_OPEN, kernel, iterations=10)
opneing = np.uint8(opening)
cv.imshow("Opening", opening)

opening_gray = cv.cvtColor(opening, cv.COLOR_BGR2GRAY)

sure_bg = cv.dilate(opening, kernel, iterations=2)
cv.imshow("Sure_bg", sure_bg)

distance_transform = cv.distanceTransform(opening_gray, cv.DIST_L2, 5)
normalized_distance = exposure.rescale_intensity(distance_transform, out_range=(0, 255))
normalized_distance = normalized_distance.astype(np.uint8)
cv.imshow("Normalized Distance", normalized_distance)
#cv.imshow("Distance Transform", distance_transform)
#cv.waitKey(0)

ret, sure_fg = cv.threshold(normalized_distance, 0.5 * normalized_distance.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
sure_fg = cv.dilate(sure_fg, kernel, iterations=4)
sure_bg = cv.cvtColor(sure_bg, cv.COLOR_BGR2GRAY)
print(sure_fg.shape)
print(sure_bg.shape)
unknown = cv.subtract(sure_bg, sure_fg)
cv.imshow("Unknown", unknown)
cv.imshow("Sure_fg", sure_fg)
cv.waitKey(0)

print(image2.shape)
ret, markers = cv.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0
markers = markers.astype(np.int32)
cv.waitKey(0)

markers = cv.watershed(image2, markers)
image2[markers == -1] = [255, 0, 0]
cv.imshow("Image2", image2)
markers = markers.astype(np.uint8)
cv.imshow("Markers", markers)
cv.waitKey(0)
mask = np.zeros_like(image2)

for i in range(len(markers)):
    if i == 1:
        continue
    color = np.random.randint(0, 255, 3)
    mask[markers == i] = color
    cv.imshow("Mask", mask)


cv.destroyAllWindows()
cv.waitKey(0)
ellipse_img = np.zeros_like(image)
roi = []
ellipsis = []
i = 1
for cnt in contours2:
    ellipse = cv.fitEllipse(cnt)
    
    # Convert ellipse coordinates to integers
    x_coord = int(ellipse[0][0])
    y_coord = int(ellipse[0][1])
    #print(x_coord, y_coord)
    #print(ellipse[0])
    # Get the size of the text
    text_size, _ = cv.getTextSize(str(i), cv.FONT_HERSHEY_SIMPLEX, 1, 2)

    inbound_x = max(0, min(x_coord, img_w - text_size[0]))
    inbound_y = max(0, min(y_coord, img_h))
    cv.putText(image2, str(i), (inbound_x, inbound_y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    ellipsis.append(ellipse)
    cv.ellipse(image2, ellipse, (0, 255, 0), 2)
    i += 1

    x, y, w, h = cv.boundingRect(cnt)
    #cv.rectangle(image2, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi_x = max(0, x-10)
    roi_y = max(0, y-10)
    roi_w = min(image.shape[1]- roi_x, w+20)
    roi_h = min(image.shape[0]- roi_y, h+20)
    roi.append(image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w])



for img in roi:
    cv.destroyAllWindows()
    cv.imshow("ROI", img)
    cv.waitKey(0)

#print("Number of rocks: ", len(roi))
#rint(len(contours2))
cv.imshow("markcont", mask) 
cv.imshow("Image", image)
cv.imshow("Image2", image2)
cv.waitKey(0)
cv.destroyAllWindows()
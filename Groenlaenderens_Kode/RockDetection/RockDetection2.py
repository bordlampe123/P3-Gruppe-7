import cv2 as cv
import numpy as np
import skimage.exposure as exposure

#Load image and get image dimensions
image = cv.imread("Groenlaenderens_Kode/RockDetection/Billeder/image_7.jpg")
image = cv.resize(image, (1280, 720))
image2 = image.copy()
image3 = image.copy()
img_h, img_w = image.shape[:2]

def Preproccesing(image, threshold):
    # Convert to HSV and split into channels
    brightened = cv.add(image, np.array([40.0]))
    HSVImage = cv.cvtColor(brightened, cv.COLOR_BGR2HSV)
    H, S, V = cv.split(HSVImage) 
    cv.imshow("H", H)
    cv.imshow("S", S)
    cv.imshow("V", V)
    cv.waitKey(0)
    # Thresholding the saturation channel,
    thresholded = cv.threshold(S, threshold, 255, cv.THRESH_BINARY_INV)[1]
    cv.imshow("Thresholded", thresholded)
    cv.waitKey(0)
    
    dilated = cv.dilate(thresholded, (3, 3), iterations=2)
    cv.imshow("Dilated", dilated)
    cv.waitKey(0)
    contours = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    contours = [cnt for cnt in contours if cv.contourArea(cnt) > 1000]
    contour_img = np.zeros_like(image)
    cv.drawContours(contour_img, contours, -1, (255, 255, 255), -1)
    cv.imshow("Contour_img", contour_img)
    cv.waitKey(0)
    return contours, contour_img

def watershed(image, x):
    contour_img = x
    #checking if image contains anything
    if np.sum(image) == 0:
        return []
    else:
        #Morphological opening
        kernel = np.ones((3, 3), np.uint8)
        opening = cv.morphologyEx(image, cv.MORPH_OPEN, kernel, iterations=10) #opening the image = erosion followed by dilation
        opening = np.uint8(opening)
        opening_gray = cv.cvtColor(opening, cv.COLOR_BGR2GRAY)
        cv.imshow("Opening", opening)
        cv.waitKey(0)

        #Finding sure background and foreground through dilation and distance transform thresholding
        sure_bg = cv.dilate(opening, kernel, iterations=2)
        cv.imshow("Sure_bg", sure_bg)
        cv.waitKey(0)
        distance_transform = cv.distanceTransform(opening_gray, cv.DIST_L2, 5)
        normalized_distance = exposure.rescale_intensity(distance_transform, out_range=(0, 255))
        normalized_distance = normalized_distance.astype(np.uint8)
        _, distThres = cv.threshold(distance_transform, 30, 255, cv.THRESH_BINARY)
        sure_bg = cv.cvtColor(sure_bg, cv.COLOR_BGR2GRAY)
        distThres = np.uint8(distThres)

        cv.imshow("Normalized Distance", normalized_distance)
        #print("Sure_bg", sure_bg.shape)
        #print("distThres", distThres.shape)
        cv.waitKey(0)
        #Finding unknown region
        unknown = cv.subtract(sure_bg, distThres)
        #cv.imshow("sure_bg", sure_bg)
        cv.imshow("distThres", distThres)
        cv.imshow("Unknown", unknown)
        #cv.waitKey(0)

        #Labeling the markers, 
        labels = cv.connectedComponents(distThres, connectivity=8, ltype=cv.CV_32S)[1] #Markerer baggrunden med 0, og markerer de forskellige objekter med 1, 2, 3 osv. Connected components = pixel i omkreds med samme værdi
        labels = labels + 1

        #Marking the unknown region with zero
        labels[unknown == 255] = -1

        mask = np.zeros_like(image)
        mask2 = np.zeros_like(image)
        #Watershed
        labels = cv.watershed(contour_img, labels)
        mask[labels == -1] = [255, 0, 0]
        cv.imshow("Mask", mask)
        uniqueLabels = np.unique(labels)
        print(uniqueLabels)
        LabelContours = []
        for label in uniqueLabels:
            if label == -1 or label == 1:
                continue
            labelMask = np.zeros_like(image)
            labelMask[labels == label] = 255
            labelMask = cv.cvtColor(labelMask, cv.COLOR_BGR2GRAY)
            contours = cv.findContours(labelMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
            LabelContours.append(contours)
        return LabelContours
        

def main(image, threshold):
    #Preproccesing
    contours, contour_img = Preproccesing(image, threshold)
    FinalImg = np.zeros_like(image)
    #Creating lists for sorting contours, and establishing thresholds
    InBoundRock = []
    OutBoundRock = []
    SortHull = []
    SingleRock = []
    MultipleRocks = []
    SolidityThres = 0.88
    EllipseThres = 0.80

    #Sorting contours based on convex and ellipse solidity, and based on image boundaries
    for cnt in contours:
        SortingConvexHull = cv.convexHull(cnt)
        SortingEllipse = cv.fitEllipse(SortingConvexHull)
        Area = cv.contourArea(cnt)
        ConvexHullArea = cv.contourArea(SortingConvexHull)
        EllipseArea = (SortingEllipse[1][0]/2)*(SortingEllipse[1][1]/2)*np.pi
        Solidity = float(Area)/ConvexHullArea
        SolidityEllipse = float(Area)/EllipseArea
        print(Solidity)
        print(SolidityEllipse)
        cv.ellipse(image2, SortingEllipse, (0, 255, 0), 2)
        cv.imshow("Image2", image2)
        cv.waitKey(0)
        if 15 < SortingEllipse[0][0] < img_w-15 and 15 < SortingEllipse[0][1] < img_h-15:
            InBoundRock.append(SortingEllipse)
            SortHull.append(SortingConvexHull)
            if Solidity < SolidityThres and SolidityEllipse < EllipseThres:
                MultipleRocks.append(cnt)
            else:
                SingleRock.append(cnt)
        else:
            OutBoundRock.append(SortingEllipse)


    MultipleRockImg = np.zeros_like(image)
    SingleRockImg = np.zeros_like(image)

    cv.drawContours(MultipleRockImg, MultipleRocks, -1, (255, 255, 255), -1)
    cv.drawContours(SingleRockImg, SingleRock, -1, (255, 255, 255), -1)

    cv.imshow("MultipleRockImg", MultipleRockImg)
    cv.imshow("SingleRockImg", SingleRockImg)
    cv.waitKey(0)

    LabelContours = watershed(MultipleRockImg, contour_img)
    AllEllipse = []
    AllContours = []

    print(FinalImg.shape)
    FinalImg[SingleRockImg == 255] = 255
    AllContours.extend(SingleRock)

    for contours in LabelContours:
        AllContours.extend(contours)
        cv.drawContours(FinalImg, contours, -1, (255, 255, 255), -1)
        
    for cnt in AllContours:
        ellipse = cv.fitEllipse(cnt)
        #print(ellipse)
        AllEllipse.append(ellipse)
        cv.ellipse(image3, ellipse, (0, 255, 0), 2)

    #Sorting ellipses based on longest axis
    for ellipse in AllEllipse:
        SortedEllipse = sorted(AllEllipse, key=lambda x: max(x[1]), reverse=True)
        cv.circle(image3, (int(ellipse[0][0]), int(ellipse[0][1])), 2, (0, 0, 255), -1)
    cv.ellipse(image3, SortedEllipse[1], (0, 0, 255), 2)
    cv.imshow("Image2", image2)
    cv.imshow("Image3", image3)
    cv.imshow("FinalImg", FinalImg)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return SortedEllipse

def GetList(x,y):
    if x == 1 and y == 0:
        return SortedEllipse[0]
    if y == 1:
        return SortedEllipse[x]


SortedEllipse = main(image, 45)
print(SortedEllipse)
#print(GetList(2,1))




#preprocessing(Input Image, threshold value)
    #Convert to HSV and split into channels
    #Thresholding the saturation channel
    #Dilating the thresholded image to ensure that the rock is mostly complete and not broken up
    #Finding contours in the dilated image
    #Sorting contours based on size and drawing the small contours on a mask
    #Subtracting the small contours from the thresholded image


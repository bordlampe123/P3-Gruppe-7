import numpy as np
import cv2 as cv

SBS = 100
iterations = int(500/SBS)

def gembillede(input, x, y):
    outputimg = input[x*SBS:(x+1)*SBS, y*SBS:(y+1)*SBS]
    return outputimg

image = cv.imread('Vores_MiniProjekt\\1.jpg')

# Create a dictionary to store the sub-images
sub_images = {}
billeder = []
for i in range(iterations):
    for j in range(iterations):
        # Generate a variable name based on i and j
        variable_name = f"billed_{j}_{i}"
        
        # Store the sub-image in the dictionary with the variable name as the key
        sub_images[variable_name] = gembillede(image, i, j)
        

# Now you have a dictionary where keys are variable names, and values are sub-images
# You can access them like this:
billeder = [[sub_images["billed_0_0"], sub_images["billed_0_1"], sub_images["billed_0_2"], sub_images["billed_0_3"], sub_images["billed_0_4"]],
            [sub_images["billed_1_0"], sub_images["billed_1_1"], sub_images["billed_1_2"], sub_images["billed_1_3"], sub_images["billed_1_4"]],
            [sub_images["billed_2_0"], sub_images["billed_2_1"], sub_images["billed_2_2"], sub_images["billed_2_3"], sub_images["billed_2_4"]],
            [sub_images["billed_3_0"], sub_images["billed_3_1"], sub_images["billed_3_2"], sub_images["billed_3_3"], sub_images["billed_3_4"]],
            [sub_images["billed_4_0"], sub_images["billed_4_1"], sub_images["billed_4_2"], sub_images["billed_4_3"], sub_images["billed_4_4"]]]





cv.imshow("Sub_Image", billeder[0][3])

# Wait for a key press and close all OpenCV windows
cv.imshow("org",image)

def prominent(img):
    data = np.reshape(img, (-1,3))
    #print(data.shape)
    data = np.float32(data)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv.KMEANS_PP_CENTERS
    compactness,labels,centers = cv.kmeans(data,1,None,criteria,10,flags)

    #print('Dominant color is: bgr({})'.format(centers[0].astype(np.int32)))
    return centers[0].astype('uint8')

sub_farver = {}


Farveholder = [255,255,255]
template = np.zeros_like(image)
y_1 = -SBS
y_2 = 0
for i in range(iterations):
    x_1 = 0
    x_2 = SBS
    y_1 = y_1 + SBS
    y_2 = y_2 + SBS
    for j in range(iterations):
        Farve = f"farver_{j}_{i}"
        #print(prominent(billeder[j][i]))
        # Store the sub-image in the dictionary with the variable name as the key
        sub_farver[Farve] = prominent(billeder[j][i])
        b = int(prominent(billeder[j][i])[0])
        g = int(prominent(billeder[j][i])[1])
        r = int(prominent(billeder[j][i])[2])
        print (j,i,b,g,r)
        template = cv.rectangle(template, (x_1, y_1), (x_2, y_2), (b, g, r), -1)
        print(x_1,x_2,y_1,y_2)
        x_1 = x_1 + SBS
        x_2 = x_2 + SBS
        

farver = np.array([  [sub_farver["farver_0_0"], sub_farver["farver_0_1"], sub_farver["farver_0_2"], sub_farver["farver_0_3"], sub_farver["farver_0_4"]],
                        [sub_farver["farver_1_0"], sub_farver["farver_1_1"], sub_farver["farver_1_2"], sub_farver["farver_1_3"], sub_farver["farver_1_4"]],
                        [sub_farver["farver_2_0"], sub_farver["farver_2_1"], sub_farver["farver_2_2"], sub_farver["farver_2_3"], sub_farver["farver_2_4"]],
                        [sub_farver["farver_3_0"], sub_farver["farver_3_1"], sub_farver["farver_3_2"], sub_farver["farver_3_3"], sub_farver["farver_3_4"]],
                        [sub_farver["farver_4_0"], sub_farver["farver_4_1"], sub_farver["farver_4_2"], sub_farver["farver_4_3"], sub_farver["farver_4_4"]]])







cv.imshow("nyt", template)

cv.waitKey(0)
cv.destroyAllWindows()



import numpy as np
import cv2 as cv

SBS = 100
iterations = int(500/SBS)
image = cv.imread('Vores_MiniProjekt\\1.jpg')
print("iterations", iterations)
def gembillede(input, x, y):
    outputimg = input[x*SBS:(x+1)*SBS, y*SBS:(y+1)*SBS]
    return outputimg

list = []
for i in range(iterations):
    for j in range(iterations):
        sub_image = gembillede(image, i, j)
        list.append(sub_image)

cv.imshow("test", list[2])

#Gul - Mark (7,155,171)
#grøn - eng (21,162,111)
#mørkegrøn - skov  (37,52,38)
#blå - vand  (192,99,14)
#brun - vissen  (43,95,108)
#sort - mine (0, 0, 0) 
#Anden brun Bordplade (19, 92, 124)

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

BGR = []
for i in range(iterations*iterations):
    BGR.append([int(prominent(list[i])[0]), int(prominent(list[i])[1]), int(prominent(list[i])[2])])

template = np.zeros_like(image)
template_Improved = np.zeros_like(image)
c = 0
y_1 = -SBS
y_2 = 0
for i in range(iterations):
    x_1 = 0
    x_2 = SBS
    y_1 = y_1 + SBS
    y_2 = y_2 + SBS
    for j in range(iterations):
        template = cv.rectangle(template, (x_1, y_1), (x_2, y_2), tuple(BGR[c]), -1)
        print("Farve i ", c, " = ", BGR[c])

        x_1 = x_1 + SBS
        x_2 = x_2 + SBS
        c = c+1
            
temp2 = np.zeros_like(template)
temp2 = sub_image[4][0]
print("shape", template.shape)
print("shape", sub_image.shape)
cv.imshow("nyt", template)
#cv.imshow("nyt lille", temp2)

cv.waitKey(0)
cv.destroyAllWindows()



#opencv testing

import numpy as np
import cv2 as cv

Bayer_input = np.array([[100,10,110,11],[9,50,8,49],[105,12,112,9],[14,52,15,54]])

RGB = np.zeros((3,3,3),dtype=np.uint8)
print(RGB)

def RGB_B(x,y):
    RGB[x,y,0] = Bayer_input[x+1, y+1]
    RGB[x,y,1] = Bayer_input[x, y+1]
    RGB[x,y,2] = Bayer_input[x, y]

def RGB_GB(x,y):
    RGB[x,y,0] = Bayer_input[x+1, y]
    RGB[x,y,1] = Bayer_input[x, y]
    RGB[x,y,2] = Bayer_input[x, y-1]

def RGB_GR(x,y):
    RGB[x,y,0] = Bayer_input[x, y+1]
    RGB[x,y,1] = Bayer_input[x, y]
    RGB[x,y,2] = Bayer_input[x-1, y]

def RGB_R(x,y):
    RGB[x,y,0] = Bayer_input[x, y]
    RGB[x,y,1] = Bayer_input[x, y-1]
    RGB[x,y,2] = Bayer_input[x-1, y-1]


def Demosiac(Matrix):
    for i in range(3):
        for j in range(3):
            if i % 2 == 0 and j % 2 == 0:
                RGB_B(i,j)
            elif  i % 2 == 0 and j % 2 == 1:
                RGB_GB(i,j)
            elif i % 2 == 1 and j % 2 == 0:
                RGB_GR(i,j)
            else:
                RGB_R(i,j)
    print(RGB)
    cv.imshow('RGB',RGB)
    cv.imwrite('RGB.png',RGB)
    cv.waitKey(0)
    


Demosiac(Bayer_input)





            
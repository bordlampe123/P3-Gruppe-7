#opencv testing

#Importing the libraries
import numpy as np
import cv2 as cv

Bayer_input = np.array([[100,10,110,11],[9,50,8,49],[105,12,112,9],[14,52,15,54]])#defining the Bayer input

RGB = np.zeros((3,3,3),dtype=np.uint8)#defining the RGB output
print(RGB)

def RGB_B(x,y):#defining the RGB output for the blue pixels
    RGB[x,y,0] = Bayer_input[x+1, y+1]
    RGB[x,y,1] = Bayer_input[x, y+1]
    RGB[x,y,2] = Bayer_input[x, y]

def RGB_GB(x,y):#defining the RGB output for the green-blue pixels
    RGB[x,y,0] = Bayer_input[x+1, y]
    RGB[x,y,1] = Bayer_input[x, y]
    RGB[x,y,2] = Bayer_input[x, y-1]

def RGB_GR(x,y):#defining the RGB output for the green-red pixels
    RGB[x,y,0] = Bayer_input[x, y+1]
    RGB[x,y,1] = Bayer_input[x, y]
    RGB[x,y,2] = Bayer_input[x-1, y]

def RGB_R(x,y):#defining the RGB output for the red pixels
    RGB[x,y,0] = Bayer_input[x, y]
    RGB[x,y,1] = Bayer_input[x, y-1]
    RGB[x,y,2] = Bayer_input[x-1, y-1]


def Demosiac(Matrix):#defining the function that creates the RGB output
    for i in range(3):#for loop that runs through the rows and columns of the RGB output
        for j in range(3):
            if i % 2 == 0 and j % 2 == 0:#if statement that checks if the pixel is blue
                RGB_B(i,j)
            elif  i % 2 == 0 and j % 2 == 1:#if statement that checks if the pixel is green-blue
                RGB_GB(i,j)
            elif i % 2 == 1 and j % 2 == 0:#if statement that checks if the pixel is green-red
                RGB_GR(i,j)
            else:
                RGB_R(i,j)#if statement that checks if the pixel is red
    print(RGB)
    cv.imshow('RGB',RGB)#showing the RGB output
    cv.imwrite('RGB.png',RGB)#saving the RGB output
    cv.waitKey(0)#waiting for a key to be pressed
    

Demosiac(Bayer_input)





            
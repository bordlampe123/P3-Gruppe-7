import cv2 as cv
import numpy as np

RGB = np.zeros((3,3,3), dtype=int)

print(RGB)

BInput = np.array([[100,10,110,11],[9,50,8,49],[105,12,112,9],[14,52,15,54]])

def RGB_B(x,y):
    RGB[x,y,0] = BInput[x+1,y+1]
    RGB[x,y,1] = BInput[x,y+1]
    RGB[x,y,2] = BInput[x,y]

def RGB_GB(x,y):
    RGB[x,y,0] = BInput[x+1,y]
    RGB[x,y,1] = BInput[x,y]
    RGB[x,y,2] = BInput[x,y-1]

def RGB_GR(x,y):
    RGB[x,y,0] = BInput[x,y+1]
    RGB[x,y,1] = BInput[x,y]
    RGB[x,y,2] = BInput[x-1,y]

def RGB_R(x,y):
    RGB[x,y,0] = BInput[x,y]
    RGB[x,y,1] = BInput[x,y-1]
    RGB[x,y,2] = BInput[x-1,y-1]

def plade(Matrix):
    for i in range(3):
        for j in range(3):
            if i%2 == 0 and j%2 == 0:
                RGB_B(i,j)
            elif i%2 == 0 and j%2 != 0:
                RGB_GB(i,j)
            elif i%2 != 0 and j%2 == 0:
                RGB_GR(i,j)
            elif i%2 != 0 and j%2 != 0:
                RGB_R(i,j)
    print(RGB)

plade(BInput)




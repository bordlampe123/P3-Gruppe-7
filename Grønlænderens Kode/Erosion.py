import cv2 as cv
import numpy as np
import math

#defining input
input = cv.imread("C:/Users/minik/Desktop/VSCode/shapes.png")

#converting to binary
input = cv.cvtColor(input, cv.COLOR_BGR2GRAY)


height = input.shape[0]
width = input.shape[1]

output = np.zeros((height, width), np.uint8)

#defining erosion function
def erosion(input, kernelSize):
    #defining kernel
    kernel = np.ones((kernelSize, kernelSize), np.uint8)
    
    kHeight, kWidth = kernel.shape

    for i in range(height):
        for j in range(width):
            if i + kHeight <= height and j + kWidth <= width:
                region = input[i:i+kHeight, j:j+kWidth]
                #if the region is equal to the kernel, then the pixel is set to 1
                if np.array_equal(region, kernel):
                    output[i,j] = 255
                else:
                    output[i,j] = 0
    return output


erosion(input, 3)
cv.imshow('input', input)
cv.imshow('output', output)
cv.waitKey(0)
cv.destroyAllWindows()
    



import cv2
import numpy as np
import math

input = cv2.imread("C:/Users/minik/Desktop/VSCode/55.jpg")#defining the input image

def meanFilter(input, kernelSize):

    output = np.zeros((input.shape[0],input.shape[1],3), dtype=np.uint8) #defining the output image
    
    kernel = np.ones((kernelSize, kernelSize), dtype=int)/(kernelSize**2) #defining the kernel, creates a kernel consisting of 1/kernelsize^2
    kernelRadius = kernelSize // 2 #defining the radius of the kernel, which is used to ensure that the kernel is within the boundaries of the input
    
    for y in range(kernelRadius, input.shape[0]-kernelRadius): #for loop that runs through the rows and columns of the input image
        for x in range(kernelRadius, input.shape[1]-kernelRadius): #for loop that runs through the pixels of the input image
            for color in range(3): #for loop that runs through the colors of the input image, so the different color channels
                sum = 0
                for ky in range(kernel.shape[0]):
                    for kx in range(kernel.shape[1]):
                        sum += input[y+ky-kernelRadius, x+kx-kernelRadius, color] * kernel[ky, kx]
                output[y, x, color] = sum
    return output
                
output = meanFilter(input, 10)
cv2.imshow('input',input)
cv2.imshow('output',output)
cv2.waitKey(0)

import cv2
import numpy as np
import math

input = cv2.imread("C:/Users/minik/Desktop/VSCode/lion.jpg")#defining the input image

def meanFilter(input, kernelSize):
    output = [] #defining the output image
    kernel = np.ones((kernelSize, kernelSize), dtype=int)/(kernelSize**2) #defining the kernel, creates a kernel consisting of 1/kernelsize^2
    kernelRadius = kernelSize // 2 #defining the radius of the kernel, which is used later to ensure that the kernel is within the boundaries of the input
    for y, row in enumerate(input): #for loop that runs through the rows and columns of the input image
        for x, pixel in enumerate(row): #for loop that runs through the pixels of the input image
            for color in range(3): #for loop that runs through the colors of the input image, so the different color channels
                accumulator = 0
                for ky in range(kernelSize): #for loop that runs through the rows and columns of the kernel
                    for kx in range(kernelSize): #for loop that runs through the rows and columns of the kernel
                        pixelX = x + kx - kernelRadius
                        pixelY = y + ky - kernelRadius
                        if pixelX 
                output[y, x, color] = accumulator
    return output

                
output = meanFilter(input, 7)
cv2.imshow('input',input)
cv2.imshow('output',output)
cv2.waitKey(0)

import cv2
import numpy as np

input = cv2.imread("C:/Users/minik/Desktop/VSCode/lion.jpg")#defining the input image
output = np.zeros(input.shape, dtype=input.dtype)#defining the output image

for y, row in enumerate(input):#for loop that runs through the rows and columns of the input image
    for x, pixel in enumerate(row):#for loop that runs through the pixels of the input image
        value = np.max(pixel)
        saturation = 0
        if value != 0:
            saturation = (value-np.min(pixel))/value
        hue = 0
        h_denominator = value - np.min(pixel)#defining the denominator of the hue formula
        blue, green, red = [int(color) for color in pixel]#defining the blue, green and red values of the pixel
        if h_denominator != 0:
            if value == red and green >= blue:
                hue = 60 * (green - blue) / h_denominator
            elif value == red and green:
                hue = 60 * ((blue - red) / h_denominator + 2)
            elif value == blue:
                hue = 60 * ((red - green) / h_denominator + 4)
            elif value == red and green < blue:
                hue = 60 * ((red - blue) / h_denominator + 5)
        hue /= 2
        output[y, x] = [hue, saturation, value]    

cv2.imshow('input',input)
cv2.imshow('output',output)
cv2.imshow('Hue', output[:,:,0])
cv2.waitKey(0)



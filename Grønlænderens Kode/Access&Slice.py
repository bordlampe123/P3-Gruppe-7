#opencv testing

import numpy as np
import cv2 as cv

goofy = cv.imread("C:/Users/NZXT/Pictures/goofy.jpg")

#numpy functions
print(f"Type of variable: {type(goofy)}")#what type of thing are we dealing with
print(f"Type of data in array: {goofy.dtype}") #uint8 = unsigned 8 bit integer, (0-255), no negatives,  #.dtype = datatype, what data are we working with
print(f"Shape of array: {goofy.shape}") #size of array?


#print af pixel værdier er omvendt, B, G, R
print("sådan der")


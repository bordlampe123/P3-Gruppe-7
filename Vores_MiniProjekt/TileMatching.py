import math
import cv2 as cv
import numpy as np
import time

#inputs the needed images
img = cv.imread("Vores_MiniProjekt/1.jpg", cv.IMREAD_COLOR) #defining the input image
grid = np.zeros((5,5), dtype= str) #defining the grid
img_1_expected_grid= np.array([["M", "S", "F", "F", "F"],
                        ["M", "F", "F", "F", "M"],
                        ["M", "T", "C", "F", "M"],
                        ["M", "T", "S", "M", "M"],
                        ["F", "S", "S", "M", "W"]]) #defining the expected grid

def matrix(output,type,bol): #defining the function for splitting the image into a grid and checks if each is below the treshhold
    output_crop = output
    height, width = output.shape
    treshhold = 180

    # Number of pieces Horizontally 
    W_SIZE  = 5 
    # Number of pieces Vertically to each Horizontal  
    H_SIZE = 5
    
    for ih in range(H_SIZE ):
        for iw in range(W_SIZE ):
            x = width/W_SIZE * iw 
            y = height/H_SIZE * ih
            h = (height / H_SIZE)
            w = (width / W_SIZE )
            output = output[int(y):int(y+h), int(x):int(x+w)]
            if bol == True:
                print(ih, iw)
                print(np.average(output))
            if np.average(output) < treshhold:
                grid[ih][iw] = type
            NAME = str(time.time()) 
            #cv.imshow("Output Images/" + str(ih)+str(iw) +  ".png",output)
            output = output_crop

def forrest_template(img,bol): #defining the function for detecting the forrest tiles in the image
    template = cv.imread("Vores_MiniProjekt/templates/tree.png", cv.IMREAD_COLOR)
    res = cv.matchTemplate(img, template, cv.TM_SQDIFF_NORMED)
    threshold = 0.8
    output = res.copy()
    for i in range(len(res)):
        for j in range(len(res[i])):
            if output[i][j] < threshold:
                output[i][j] = 0
            else:
                output[i][j] = 255
    if bol == True:
        cv.imshow("F res", res)
        cv.imshow("Forrest output", output)
    matrix(output, "F",bol)

def sea_template(img,bol): #defining the function for detecting the sea tiles in the image
    template = cv.imread("Vores_MiniProjekt/templates/sea.png", cv.IMREAD_COLOR)
    res = cv.matchTemplate(img, template, cv.TM_SQDIFF_NORMED)
    threshold = 0.7
    output = res.copy()
    for i in range(len(res)):
        for j in range(len(res[i])):
            if output[i][j] < threshold:
                output[i][j] = 0
            else:
                output[i][j] = 255
    if bol == True:
        cv.imshow("S res", res)
        cv.imshow("Sea output", output)
    matrix(output, "S",bol)

def meadow_template(img,bol): #defining the function for detecting the meadow tiles in the image
    template = cv.imread("Vores_MiniProjekt/templates/meadow.png", cv.IMREAD_COLOR)
    res = cv.matchTemplate(img, template, cv.TM_SQDIFF_NORMED)
    threshold = 0.15
    output = res.copy()
    for i in range(len(res)):
        for j in range(len(res[i])):
            if output[i][j] < threshold:
                output[i][j] = 0
            else:
                output[i][j] = 255
    if bol == True:
        cv.imshow("M res", res)
        cv.imshow("meadow output", output)
    matrix(output, "M",bol)

def wheat_template(img,bol): #defining the function for detecting the wheat tiles in the image
    template = cv.imread("Vores_MiniProjekt/templates/wheat.png", cv.IMREAD_COLOR)
    res = cv.matchTemplate(img, template, cv.TM_SQDIFF_NORMED)
    threshold = 0.1
    output = res.copy()
    for i in range(len(res)):
        for j in range(len(res[i])):
            if output[i][j] < threshold:
                output[i][j] = 0
            else:
                output[i][j] = 255
    if bol == True:
        cv.imshow("W res", res)
        cv.imshow("wheat output", output)
    matrix(output, "W",bol)

def tundra_template(img,bol): #defining the function for detecting the tundra tiles in the image
    template = cv.imread("Vores_MiniProjekt/templates/tundra.png", cv.IMREAD_COLOR)
    res = cv.matchTemplate(img, template, cv.TM_SQDIFF_NORMED)
    threshold = 0.2
    output = res.copy()
    for i in range(len(res)):
        for j in range(len(res[i])):
            if output[i][j] < threshold:
                output[i][j] = 0
            else:
                output[i][j] = 255
    if bol == True:
        cv.imshow("T res", res)
        cv.imshow("Tundra output", output)
    matrix(output, "T",bol)

def run_templates(img,bol):
    forrest_template(img,bol)
    sea_template(img,bol)
    meadow_template(img,bol)
    wheat_template(img,bol)

    cv.imshow("img", img)
    print("Grid:")
    print(grid)
    print("img 1 Expected grid:")
    print(img_1_expected_grid)

#run_templates(img,True)
#run_templates(img,False)
tundra_template(img,True)
print(grid)

cv.waitKey(0)
cv.destroyAllWindows()
import cv2 as cv
import numpy as np
import math

img = np.array([[0,0,0,255,255],
                [0,0,0,0,255],
                [0,0,0,0,255],
                [0,0,255,255,0],
                [0,0,255,255,0]])
               
def grassfire(img,coord,id,type):
    y,x = coord
    burn_queue = []  
    group = []
    
    if img[y,x] != type:
        burn_queue.append((y,x))
        
    while len(burn_queue) > 0:
        current = burn_queue.pop()
        y,x = current
        group.append((y,x))
        img[y,x] = id
        if x+1 < img.shape[1] and img[y,x+1] == type:
            burn_queue.append((y,x+1))
        if x > 0 and img[y,x-1] == type:
            burn_queue.append((y,x-1))
        if y+1 < img.shape[0] and img[y+1,x] == type:
            burn_queue.append((y+1,x))
        if y > 0 and img[y-1,x] == type:
            burn_queue.append((y-1,x))
        print(img)
        print(blobs)
        if len(burn_queue) ==0:
            blobs.append(group)
            return id+1
    return id

blobs = []
type = 0
id = 1

    
for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        next_id = grassfire(img, (y, x),id,type)
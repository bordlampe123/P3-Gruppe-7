import cv2 as cv
import numpy as np
import math

img = np.array([[11,13,12,12,12],
                 [11,12,12,12,11],
                 [11,14,17,12,11],
                 [11,14,13,11,11],
                 [12,13,13,11,15]])
               
crown = np.array([[0,0,0,0,0],
                   [0,0,0,1,0],
                   [0,1,0,0,0],
                   [0,2,0,2,1],
                   [0,1,0,1,0]])
                  

def grassfire(img,coord,id):
    y,x = coord
    burn_queue = []  
    group = []
    
    if (y,x) in list:
        return id
    else:
        type = img[y,x]
        burn_queue.append((y,x))
        
    while len(burn_queue) > 0:
        current = burn_queue.pop()
        y,x = current
        if current not in group:
            group.append((y,x))
            list.append((y,x))
            img[y,x] = id
            if x+1 < img.shape[1] and img[y,x+1] == type:
                burn_queue.append((y,x+1))
            if x > 0 and img[y,x-1] == type:
                burn_queue.append((y,x-1))
            if y+1 < img.shape[0] and img[y+1,x] == type:
                burn_queue.append((y+1,x))
            if y > 0 and img[y-1,x] == type:
                burn_queue.append((y-1,x))
        
        if len(burn_queue) ==0:
            blobs.append(group)
            print(img)
            return id+1
    return id

def crownCounter(crown,y,x):
    krone = crown[y,x]
    for z in range(len(blobs)):
        for w in range(len(blobs[z])):
            try:
                if blobs[z][w][0] == y and blobs[z][w][1] == x:
                    blobs[z][w] = krone
                    return
            except Exception:
                pass           

Nid = 1
blobs = []
list = []
sumP = 0  

for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        Nid = grassfire(img, (y, x),Nid)
for y in range(crown.shape[0]):
        for x in range(crown.shape[1]):
            crownCounter(crown,y,x)

for x in range(len(blobs)):
    sumP = sumP + len(blobs[x])*sum(blobs[x])
    print(len(blobs[x])*sum(blobs[x]))
print("den samlet værdi er", sumP)
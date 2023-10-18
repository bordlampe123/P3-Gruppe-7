import numpy as np

img = np.array([[0,0,0,255,255,255],
                [0,0,0,0,255,0],
                [0,0,0,0,255,0],
                [0,0,255,255,0,0],
                [0,0,255,255,0,0],
                [0,0,255,255,0,0]], dtype=np.uint8)

def grassfire(img,coord,id):
    y,x = coord

    burn_queue = []  
    
    if img[y,x] == 255:
        burn_queue.append((y,x))

    while len(burn_queue) > 0:
        current = burn_queue.pop()
        y,x = current
        img[y,x] = id
        if x+1 < img.shape[1] and img[y,x+1] == 255:
            burn_queue.append((y,x+1))
        if x > 0 and img[y,x-1] == 255:
            burn_queue.append((y,x-1))
        if y+1 < img.shape[0] and img[y+1,x] == 255:
            burn_queue.append((y+1,x))
        if y > 0 and img[y-1,x] == 255:
            burn_queue.append((y-1,x))
        print(img)

        if len(burn_queue) ==0:
            return id+1
    return id

next_id = 1

for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        next_id = grassfire(img, (y, x), next_id)
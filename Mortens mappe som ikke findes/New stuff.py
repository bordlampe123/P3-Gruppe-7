import cv2
import numpy as np
img = cv2.imread()

def convolve(img, kernel):
kernel_size = 11
Kernal = np.ones((kernel_size, kernel_size))
output = np.zeros(shape: (img.shape[0]-kernel_size, img.shape[1]-kernel_size)),dtype


for y, row in enumerate(output):
    for x, row in enumerate(row):
        slice = img[y:y+kernel_size, x:x+kernel_size]
        output[y,x] = np.sum(slice*kernel)/np.sum(kernel)


cv2.imshow()
cv2.waitKey()


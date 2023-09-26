import cv2 as cv
import numpy as np

img = cv.imread("HamDenSejer's_kode/lion.jpg")



def meanfilter(img, kernelsize):
    output = np.zeros(img.shape)
    if kernelsize % 2 == 0:
        print("Kernelsize must be odd")
        return
    kernel = np.zeros((kernelsize, kernelsize))
    kernel += 1
    kernel = kernel / (kernelsize**2)
    for x, row in enumerate(img):
        for y, pixel in enumerate(row):
            if x < kernelsize//2 or y < kernelsize//2 or x > img.shape[0] - kernelsize//2 or y > img.shape[1] - kernelsize//2:
                output[x][y] = pixel
                continue
            output[x][y] = np.sum(img[x-kernelsize//2:x+kernelsize//2+1, y-kernelsize//2:y+kernelsize//2+1] * kernel)
    return output

cv.imshow("filter",meanfilter(img, 3))

cv.imshow("image", img)
cv.waitKey(0)
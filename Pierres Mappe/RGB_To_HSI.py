import cv2
import numpy as np
import math

img = cv2.imread("Pierres Mappe/letter.png")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_hsi = np.zeros((img_rgb.shape[0], img_rgb.shape[1], 3), np.uint8)

def rgb_hsi(y, x):
    R = img_rgb[y, x, 0]
    G = img_rgb[y, x, 1]
    B = img_rgb[y, x, 2]
    if R != 0 and G != 0 and B != 0:
        thetanum = (((R-G)+(R-B))/2)/math.sqrt(((R-G)**2)+((R-B)*(G-B)))
    else:
        thetanum = 0

    theta = np.arccos(thetanum)

    print(thetanum, theta, R, G, B )
    """ theta = theta*(math.pi/180)
    if B <= G:
        H = theta
    if B > G:
        H = 360-theta
    S = 1-((3*min(R, G, B))/(R+G+B))
    I = (R+G+B)/3 """

    #return [H, S, I]

for y in range(img_rgb.shape[0]):
    for x in range(img_rgb.shape[1]):
               rgb_hsi(y, x)